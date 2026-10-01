"""Versioned deterministic replay checkpoints with explicit migrations.

Replay v3 records external kernel events with their simulation step so explicit
reasoning, connections, mechanism calls and predictive-self counterfactuals can
be replayed in the same order as the simulated experience.
"""
from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .kernel import KernelConfig, UnifiedMind


REPLAY_FORMAT_V2 = "digital-mind-replay-v2"
REPLAY_FORMAT = "digital-mind-replay-v3"
REPLAY_SCHEMA_VERSION = 3
STATE_FINGERPRINT_ALGORITHM = "digital-mind-state-v3"
PREVIOUS_STATE_FINGERPRINT_ALGORITHM = "digital-mind-state-v2"
LEGACY_STATE_FINGERPRINT_ALGORITHM = "digital-mind-state-v1"
LEGACY_REPORT_VERSIONS = ("0.2.0", "0.2.1", "0.2.2")
_VALUE_TAG = "__digital_mind_value_type__"


class ReplayMigrationRequired(ValueError):
    """The checkpoint cannot be proven compatible with the current runtime."""


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.l0")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_payload(mind, *, include_report_version, report_version=None):
    report = copy.deepcopy(mind.report())
    if report_version is not None:
        report["version"] = report_version
    if not include_report_version:
        report.pop("version", None)
    return {
        "report": report,
        "cognitive_state": mind.cognition.state,
        "self_stats": mind.cognition.self_model._stats,
        "mechanisms": mind.cognition.mechanism_document(),
        "evolution": mind.evolution.state_document(),
    }


def _raw_fingerprint(mind, *, include_report_version, report_version=None):
    digest = hashlib.sha256()
    for array in (mind.obs, mind.model.W, mind.model.W_fast, mind.self_trace):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    payload = _state_payload(
        mind,
        include_report_version=include_report_version,
        report_version=report_version,
    )
    digest.update(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def _fingerprint_json_value(value):
    if isinstance(value, np.ndarray):
        return _fingerprint_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _fingerprint_json_value(value.item())
    if isinstance(value, dict):
        return {key: _fingerprint_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fingerprint(mind, *, include_report_version, report_version=None):
    digest = hashlib.sha256()
    for array in (mind.obs, mind.model.W, mind.model.W_fast, mind.self_trace):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    payload = _fingerprint_json_value(_state_payload(
        mind,
        include_report_version=include_report_version,
        report_version=report_version,
    ))
    digest.update(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def _legacy_state_fingerprint(mind, report_version=None):
    return _raw_fingerprint(
        mind,
        include_report_version=True,
        report_version=report_version,
    )


def _previous_state_fingerprint(mind):
    return _raw_fingerprint(mind, include_report_version=False)


def state_fingerprint(mind):
    return _fingerprint(mind, include_report_version=False)


def _encode_event_value(value):
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("external journal array contains non-finite values")
        return {_VALUE_TAG: "ndarray", "shape": list(value.shape), "items": value.tolist()}
    if isinstance(value, np.generic):
        return _encode_event_value(value.item())
    if isinstance(value, tuple):
        return {_VALUE_TAG: "tuple", "items": [_encode_event_value(x) for x in value]}
    if isinstance(value, list):
        return [_encode_event_value(x) for x in value]
    if isinstance(value, dict):
        return {
            _VALUE_TAG: "dict",
            "items": [[_encode_event_value(k), _encode_event_value(v)] for k, v in value.items()],
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("external journal contains a non-finite float")
        return value
    raise ValueError(f"external journal contains unsupported value type: {type(value).__name__}")


def _decode_event_value(value):
    if isinstance(value, list):
        return [_decode_event_value(x) for x in value]
    if not isinstance(value, dict) or _VALUE_TAG not in value:
        return value
    kind = value.get(_VALUE_TAG)
    if kind == "tuple" and set(value) == {_VALUE_TAG, "items"}:
        return tuple(_decode_event_value(x) for x in value["items"])
    if kind == "dict" and set(value) == {_VALUE_TAG, "items"}:
        result = {}
        for pair in value["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid encoded dictionary entry")
            result[_decode_event_value(pair[0])] = _decode_event_value(pair[1])
        return result
    if kind == "ndarray" and set(value) == {_VALUE_TAG, "shape", "items"}:
        array = np.asarray(value["items"], dtype=float)
        shape = value["shape"]
        if (not isinstance(shape, list)
                or not all(type(x) is int and x >= 0 for x in shape)
                or tuple(shape) != array.shape
                or not np.all(np.isfinite(array))):
            raise ValueError("invalid encoded ndarray")
        return array
    raise ValueError("invalid external journal value encoding")


def _encoded_external_journal(mind):
    journal = getattr(mind, "_external_journal", [])
    if not isinstance(journal, list):
        raise ValueError("kernel external journal must be a list")
    return [_encode_event_value(copy.deepcopy(event)) for event in journal]


def migrate_replay_document(document):
    if not isinstance(document, dict):
        raise ValueError("replay document must be an object")
    replay_format = document.get("format")
    if replay_format == REPLAY_FORMAT:
        if document.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ReplayMigrationRequired("unsupported replay schema version for digital-mind-replay-v3")
        migrated = copy.deepcopy(document)
        history = migrated.get("migration_history", [])
        if not isinstance(history, list):
            raise ValueError("migration_history must be a list")
        migrated["migration_history"] = history
        migrated.setdefault("external_journal", [])
        migrated.setdefault("external_counters", {"tasks": 0, "connections": 0})
        return migrated
    if replay_format == REPLAY_FORMAT_V2:
        return {
            "format": REPLAY_FORMAT,
            "schema_version": REPLAY_SCHEMA_VERSION,
            "config": copy.deepcopy(document.get("config")),
            "steps": document.get("steps"),
            "provenance": {
                "package_version": None,
                "source_sha256": document.get("source_sha256"),
                "numpy": document.get("numpy"),
                "scipy": document.get("scipy"),
            },
            "state_fingerprint": {
                "algorithm": LEGACY_STATE_FINGERPRINT_ALGORITHM,
                "sha256": document.get("state_sha256"),
            },
            "initial_mechanisms": copy.deepcopy(document.get("initial_mechanisms")),
            "external_journal": [],
            "external_counters": {"tasks": 0, "connections": 0},
            "migration_history": [{
                "from": REPLAY_FORMAT_V2,
                "to": REPLAY_FORMAT,
                "strategy": "validated-deterministic-replay",
            }],
        }
    raise ValueError("unsupported replay format")


def _validate_state_fingerprint(mind, document):
    fingerprint = document.get("state_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("state_fingerprint must be an object")
    algorithm = fingerprint.get("algorithm")
    expected = fingerprint.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("checkpoint state fingerprint is invalid")
    if algorithm == STATE_FINGERPRINT_ALGORITHM:
        if state_fingerprint(mind) != expected:
            raise ReplayMigrationRequired(
                "replayed state does not match the saved checkpoint; a state migration is required"
            )
        return {"algorithm": algorithm, "legacy_report_version": None}
    if algorithm == PREVIOUS_STATE_FINGERPRINT_ALGORITHM:
        if _previous_state_fingerprint(mind) != expected:
            raise ReplayMigrationRequired(
                "previous replay state does not match the saved checkpoint; a state migration is required"
            )
        return {"algorithm": algorithm, "legacy_report_version": None}
    if algorithm == LEGACY_STATE_FINGERPRINT_ALGORITHM:
        versions = []
        current = mind.report().get("version")
        if isinstance(current, str):
            versions.append(current)
        versions.extend(LEGACY_REPORT_VERSIONS)
        seen = set()
        for version in versions:
            if version in seen:
                continue
            seen.add(version)
            if _legacy_state_fingerprint(mind, report_version=version) == expected:
                return {"algorithm": algorithm, "legacy_report_version": version}
        raise ReplayMigrationRequired(
            "legacy replay could not be reproduced by the current kernel; an explicit migration is required"
        )
    raise ReplayMigrationRequired(f"unsupported state fingerprint algorithm: {algorithm!r}")


def _validate_common(document):
    steps = document.get("steps")
    if type(steps) is not int or steps < 0:
        raise ValueError("replay steps must be a nonnegative integer")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("replay config must be an object")
    try:
        config = KernelConfig(**config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid replay config: {exc}") from exc
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("replay provenance must be an object")
    return steps, config, provenance


def _decode_external_journal(document, steps):
    encoded = document.get("external_journal", [])
    if not isinstance(encoded, list):
        raise ValueError("external_journal must be a list")
    events = []
    last_step = 0
    allowed = {"process", "connect", "self_counterfactual", "run_mechanism"}
    for encoded_event in encoded:
        event = _decode_event_value(encoded_event)
        if not isinstance(event, dict):
            raise ValueError("external journal event must decode to an object")
        kind, step = event.get("kind"), event.get("step")
        if kind not in allowed:
            raise ValueError("unsupported external journal event")
        if type(step) is not int or not 0 <= step <= steps or step < last_step:
            raise ValueError("external journal steps must be ordered and in range")
        last_step = step
        if kind in {"process", "connect"} and not isinstance(event.get("task"), dict):
            raise ValueError("external cognition event must contain a task object")
        if kind == "connect":
            index = event.get("connection_index")
            if type(index) is not int or index < 1:
                raise ValueError("connection event requires a positive connection_index")
        if kind == "self_counterfactual" and not isinstance(event.get("overrides"), dict):
            raise ValueError("self_counterfactual event requires overrides")
        if kind == "run_mechanism":
            if not isinstance(event.get("name"), str) or not isinstance(event.get("inputs"), dict):
                raise ValueError("run_mechanism event requires name and inputs")
        events.append(event)
    counters = document.get("external_counters", {})
    if not isinstance(counters, dict):
        raise ValueError("external_counters must be an object")
    tasks, connections = counters.get("tasks", 0), counters.get("connections", 0)
    if type(tasks) is not int or tasks < 0 or type(connections) is not int or connections < 0:
        raise ValueError("external counters must be nonnegative integers")
    return events, {"tasks": tasks, "connections": connections}


def _apply_external_event(mind, event):
    kind = event["kind"]
    if kind == "process":
        mind.cognition.process(copy.deepcopy(event["task"]))
        mind._external_tasks += 1
    elif kind == "connect":
        expected = mind._connections + 1
        if event["connection_index"] != expected:
            raise ReplayMigrationRequired("connection journal order does not match the saved checkpoint")
        mind._connections = expected
        mind.cognition.process(copy.deepcopy(event["task"]))
        mind._external_tasks += 1
    elif kind == "self_counterfactual":
        if mind._decision is None:
            raise ReplayMigrationRequired("self-counterfactual journal event has no replayed decision state")
        mind.predictive_self.counterfactual(
            mind._self_state_vector(),
            mind._self_decision_vector(),
            **copy.deepcopy(event["overrides"]),
        )
    elif kind == "run_mechanism":
        mind.cognition.run_mechanism(event["name"], **copy.deepcopy(event["inputs"]))
    else:
        raise ValueError("unsupported external journal event")
    mind._external_journal.append(copy.deepcopy(event))


def replay_document(mind):
    return {
        "format": REPLAY_FORMAT,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "config": asdict(mind.config),
        "steps": mind.fast_steps,
        "provenance": {
            "package_version": __version__,
            "source_sha256": source_fingerprint(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "state_fingerprint": {
            "algorithm": STATE_FINGERPRINT_ALGORITHM,
            "sha256": state_fingerprint(mind),
        },
        "initial_mechanisms": copy.deepcopy(mind._initial_mechanisms),
        "external_journal": _encoded_external_journal(mind),
        "external_counters": {
            "tasks": int(getattr(mind, "_external_tasks", 0)),
            "connections": int(getattr(mind, "_connections", 0)),
        },
        "migration_history": [],
    }


def restore_replay_document(raw):
    document = migrate_replay_document(raw)
    steps, config, provenance = _validate_common(document)
    events, counters = _decode_external_journal(document, steps)
    mind = UnifiedMind(config)
    initial = document.get("initial_mechanisms")
    if initial is not None:
        mind.cognition.restore_mechanism_document(initial)
        mind._initial_mechanisms = copy.deepcopy(initial)
        mind.evolution.adopt_loaded()
    for event in events:
        delta = event["step"] - mind.fast_steps
        if delta < 0:
            raise ValueError("external journal moved backwards in simulation time")
        if delta:
            mind.run(delta)
        _apply_external_event(mind, event)
    if mind.fast_steps < steps:
        mind.run(steps - mind.fast_steps)
    if mind._external_tasks != counters["tasks"]:
        raise ReplayMigrationRequired("external task count does not match replay journal")
    if mind._connections != counters["connections"]:
        raise ReplayMigrationRequired("connection count does not match replay journal")
    compatibility = _validate_state_fingerprint(mind, document)
    mind._replay_metadata = {
        "format": document["format"],
        "schema_version": document["schema_version"],
        "provenance": copy.deepcopy(provenance),
        "migration_history": copy.deepcopy(document.get("migration_history", [])),
        "compatibility": compatibility,
        "external_events": len(events),
        "current_source_sha256": source_fingerprint(),
        "current_numpy": np.__version__,
        "current_scipy": scipy.__version__,
    }
    return mind


def save_replay(mind, path):
    document = replay_document(mind)
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_replay(path):
    return restore_replay_document(json.loads(Path(path).read_text(encoding="utf-8")))


def upgrade_replay(path, destination=None):
    source = Path(path)
    target = Path(destination) if destination is not None else source
    mind = load_replay(source)
    save_replay(mind, target)
    return target
