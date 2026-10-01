"""Single-file atomic checkpoint for the complete experimental kernel state.

The checkpoint embeds a deterministic replay plus explicit, inspectable
snapshots of cognition, autobiographical memory, self-model, generated
mechanisms and autonomous-development state. The file is written with fsync
and atomic replace, and a save is rejected if the kernel or source changes
while the checkpoint is being assembled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import __version__
from .persistence import (
    load_replay,
    replay_document,
    restore_replay_document,
    source_fingerprint,
    state_fingerprint,
)


CHECKPOINT_FORMAT = "digital-mind-atomic-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION = 1
INTEGRITY_ALGORITHM = "sha256-canonical-json-v1"
_AUTO = object()


class AtomicCheckpointError(ValueError):
    pass


class CheckpointRaceError(AtomicCheckpointError):
    pass


def _json_native(value):
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise AtomicCheckpointError("checkpoint contains a non-finite ndarray")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if is_dataclass(value):
        return _json_native(asdict(value))
    if isinstance(value, deque):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_native(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AtomicCheckpointError("checkpoint contains a non-finite float")
        return value
    raise AtomicCheckpointError(
        f"checkpoint contains unsupported value type: {type(value).__name__}"
    )


def _canonical_bytes(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_document(document):
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _git_commit(root):
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha and len(env_sha) >= 7:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=True,
        )
        value = proc.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _question_document(generator):
    return _json_native({
        "feature_names": generator.feature_names,
        "residuals": list(generator.residuals),
        "features": list(generator.features),
        "timestamps": list(generator.timestamps),
        "samples_seen": generator.samples_seen,
        "questions": [vars(q) for q in generator.questions],
        "qid": generator._qid,
        "last_question_t": generator.last_question_t,
        "cooldown": generator.cooldown,
    })


def _narrative_document(narrative):
    return _json_native({
        "nodes": [vars(node) for node in narrative.nodes],
        "edges": narrative.edges,
        "theme_threshold": narrative.theme_threshold,
        "theme_centroids": narrative.theme_centroids,
        "theme_counts": dict(narrative.theme_counts),
        "next_theme": narrative._next_theme,
        "social_expectation": narrative.social_expectation,
        "social_trust": narrative.social_trust,
        "social_error_ema": narrative.social_error_ema,
    })


def _goals_document(mind):
    goals = mind.goals
    return _json_native({
        "goals": [vars(goals.goals[gid]) for gid in sorted(goals.goals)],
        "children": dict(goals.children),
        "gid": goals._gid,
        "last_state_goal_t": goals.last_state_goal_t,
        "current_goal_id": mind.current_goal.gid,
    })


def _experiment_document(mind):
    return _json_native({
        "experiments": [vars(item) for item in mind.experiments],
        "active_question": None if mind.experiment is None else mind.experiment.qid,
        "pending_treatment": mind.pending_treatment,
        "probe_steps": mind.probe_steps,
    })


def component_documents(mind):
    return {
        "cognition": _json_native({
            "state": mind.cognition.state,
            "operational_self_stats": mind.cognition.self_model._stats,
            "mechanism_calls": mind.cognition.mechanism_calls,
            "evolution": mind.evolution.state_document(),
        }),
        "memory": {
            "narrative": _narrative_document(mind.narrative),
            "goals": _goals_document(mind),
            "questions": _question_document(mind.questions),
        },
        "self_model": _json_native({
            "causal_boundary": mind.self_trace,
            "competence": mind.competence,
            "max_sensitivity": mind.max_sensitivity,
            "consolidated_self": mind.consolidated_self,
            "predictive_self": mind.predictive_self.state_document(),
        }),
        "mechanisms": _json_native(mind.cognition.mechanism_document()),
        "runtime": _json_native({
            "steps": mind.fast_steps,
            "clocks": {
                "meso": mind.meso_ticks,
                "macro": mind.macro_ticks,
                "slow": mind.slow_ticks,
                "plans": mind.plan_ticks,
            },
            "obs": mind.obs,
            "macro_action": mind.macro_action,
            "trace": list(mind.trace),
            "decision": mind._decision,
            "step_errors": mind._step_errors,
            "step_rewards": mind._step_rewards,
            "planned_goal_id": mind.planned_goal_id,
            "external_tasks": mind._external_tasks,
            "connections": mind._connections,
            "external_journal": mind._external_journal,
            "experiments": _experiment_document(mind),
        }),
    }


def _read_autodev(root, explicit):
    if explicit is not _AUTO:
        if explicit is None:
            return {
                "present": False,
                "source": "explicit-none",
                "state": None,
                "source_sha256": None,
            }, None
        if not isinstance(explicit, dict):
            raise AtomicCheckpointError("autodev_state must be an object or null")
        state = _json_native(copy.deepcopy(explicit))
        return {
            "present": True,
            "source": "explicit",
            "state": state,
            "source_sha256": _sha256_document(state),
        }, None

    path = root / "development" / "autodev_state.json"
    if not path.exists():
        return {
            "present": False,
            "source": "not-found",
            "state": None,
            "source_sha256": None,
        }, None
    raw = path.read_bytes()
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicCheckpointError(f"invalid autodev state: {exc}") from exc
    if not isinstance(state, dict):
        raise AtomicCheckpointError("autodev state must be a JSON object")
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "present": True,
        "source": "development/autodev_state.json",
        "state": _json_native(state),
        "source_sha256": digest,
    }, (path, digest)


def _assert_component_match(expected, actual):
    for name in ("cognition", "memory", "self_model", "mechanisms", "runtime"):
        if _sha256_document(expected[name]) != _sha256_document(actual[name]):
            raise AtomicCheckpointError(
                f"checkpoint round-trip changed component: {name}"
            )


def build_checkpoint_document(
    mind,
    *,
    autodev_state=_AUTO,
    repo_root=None,
    verify=True,
):
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    source_before = source_fingerprint()
    state_before = state_fingerprint(mind)
    step_before = mind.fast_steps
    journal_before = len(getattr(mind, "_external_journal", []))

    autodev, autodev_guard = _read_autodev(root, autodev_state)
    replay = replay_document(mind)
    components = component_documents(mind)

    if verify:
        restored = restore_replay_document(copy.deepcopy(replay))
        _assert_component_match(components, component_documents(restored))

    if source_fingerprint() != source_before:
        raise CheckpointRaceError("kernel source changed while checkpoint was being built")
    if state_fingerprint(mind) != state_before:
        raise CheckpointRaceError("kernel state changed while checkpoint was being built")
    if mind.fast_steps != step_before or len(getattr(mind, "_external_journal", [])) != journal_before:
        raise CheckpointRaceError("kernel timeline changed while checkpoint was being built")

    if autodev_guard is not None:
        path, expected_hash = autodev_guard
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise CheckpointRaceError("autodev state changed while checkpoint was being built")

    component_hashes = {
        name: _sha256_document(document)
        for name, document in components.items()
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": {
            "package_version": __version__,
            "git_commit": _git_commit(root),
            "source_sha256": source_before,
            "state_sha256": state_before,
        },
        "replay": replay,
        "components": components,
        "component_hashes": component_hashes,
        "autodev": autodev,
    }
    digest = _sha256_document(payload)
    return {
        **payload,
        "integrity": {
            "algorithm": INTEGRITY_ALGORITHM,
            "payload_sha256": digest,
            "checkpoint_id": digest,
        },
    }


def _atomic_json_write(path, document):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        try:
            flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
            fd = os.open(str(destination.parent), flags)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_checkpoint(
    mind,
    path,
    *,
    autodev_state=_AUTO,
    repo_root=None,
    verify=True,
):
    document = build_checkpoint_document(
        mind,
        autodev_state=autodev_state,
        repo_root=repo_root,
        verify=verify,
    )
    _atomic_json_write(path, document)
    return document["integrity"]["checkpoint_id"]


def read_checkpoint_document(path):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicCheckpointError(f"cannot read checkpoint: {exc}") from exc
    if not isinstance(document, dict) or document.get("format") != CHECKPOINT_FORMAT:
        raise AtomicCheckpointError("unsupported atomic checkpoint format")
    if document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise AtomicCheckpointError("unsupported atomic checkpoint schema version")
    integrity = document.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != INTEGRITY_ALGORITHM:
        raise AtomicCheckpointError("invalid checkpoint integrity metadata")
    expected = integrity.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AtomicCheckpointError("invalid checkpoint integrity hash")
    payload = {key: value for key, value in document.items() if key != "integrity"}
    if _sha256_document(payload) != expected:
        raise AtomicCheckpointError("atomic checkpoint integrity mismatch")
    return document


def load_checkpoint(path):
    document = read_checkpoint_document(path)
    mind = restore_replay_document(copy.deepcopy(document["replay"]))
    actual = component_documents(mind)
    expected = document.get("components")
    if not isinstance(expected, dict):
        raise AtomicCheckpointError("checkpoint components are missing")
    _assert_component_match(expected, actual)

    saved_hashes = document.get("component_hashes")
    if not isinstance(saved_hashes, dict):
        raise AtomicCheckpointError("component hashes are missing")
    for name, component in expected.items():
        if saved_hashes.get(name) != _sha256_document(component):
            raise AtomicCheckpointError(f"component hash mismatch: {name}")

    mind._atomic_checkpoint_metadata = {
        "identity": copy.deepcopy(document["identity"]),
        "integrity": copy.deepcopy(document["integrity"]),
        "component_hashes": copy.deepcopy(saved_hashes),
    }
    mind._autodev_state = copy.deepcopy(document.get("autodev"))
    return mind


def load_checkpoint_or_replay(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("format") == CHECKPOINT_FORMAT:
        return load_checkpoint(path)
    return load_replay(path)


def restore_autodev_state(checkpoint_path, destination):
    document = read_checkpoint_document(checkpoint_path)
    autodev = document.get("autodev")
    if not isinstance(autodev, dict) or not autodev.get("present"):
        raise AtomicCheckpointError("checkpoint contains no autodev state")
    state = autodev.get("state")
    if not isinstance(state, dict):
        raise AtomicCheckpointError("checkpoint autodev state is invalid")
    _atomic_json_write(destination, state)
    return Path(destination)
