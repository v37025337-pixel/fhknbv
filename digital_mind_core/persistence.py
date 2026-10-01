"""Versioned deterministic replay checkpoints with explicit migrations.

Replay files are data-only JSON.  Source/library versions are provenance, not a
blanket compatibility gate: restoration is accepted only when deterministic
replay reproduces the checkpoint's state fingerprint.  This lets non-semantic
code changes preserve continuity while still failing closed on behavioral drift.
"""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .kernel import KernelConfig, UnifiedMind


REPLAY_FORMAT_V2 = "digital-mind-replay-v2"
REPLAY_FORMAT = "digital-mind-replay-v3"
REPLAY_SCHEMA_VERSION = 3
STATE_FINGERPRINT_ALGORITHM = "digital-mind-state-v2"
LEGACY_STATE_FINGERPRINT_ALGORITHM = "digital-mind-state-v1"
LEGACY_REPORT_VERSIONS = ("0.2.0", "0.2.1", "0.2.2")


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


def _fingerprint(mind, *, include_report_version, report_version=None):
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


def _legacy_state_fingerprint(mind, report_version=None):
    """Exact v2-era fingerprint, optionally substituting the old package version."""
    return _fingerprint(
        mind,
        include_report_version=True,
        report_version=report_version,
    )


def state_fingerprint(mind):
    """Stable v3 fingerprint; package display version is intentionally excluded."""
    return _fingerprint(mind, include_report_version=False)


def migrate_replay_document(document):
    """Normalize supported historical replay schemas to the current v3 schema."""
    if not isinstance(document, dict):
        raise ValueError("replay document must be an object")

    replay_format = document.get("format")
    if replay_format == REPLAY_FORMAT:
        if document.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ReplayMigrationRequired(
                "unsupported replay schema version for digital-mind-replay-v3"
            )
        migrated = copy.deepcopy(document)
        history = migrated.get("migration_history", [])
        if not isinstance(history, list):
            raise ValueError("migration_history must be a list")
        migrated["migration_history"] = history
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
        actual = state_fingerprint(mind)
        if actual != expected:
            raise ReplayMigrationRequired(
                "replayed state does not match the saved checkpoint; "
                "a state migration is required"
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
                return {
                    "algorithm": algorithm,
                    "legacy_report_version": version,
                }
        raise ReplayMigrationRequired(
            "legacy replay could not be reproduced by the current kernel; "
            "an explicit migration is required"
        )

    raise ReplayMigrationRequired(
        f"unsupported state fingerprint algorithm: {algorithm!r}"
    )


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


def save_replay(mind, path):
    if mind._external_tasks:
        raise ValueError(
            "replay covers simulated experience; save external-task mechanisms separately"
        )
    document = {
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
        "migration_history": [],
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_replay(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    document = migrate_replay_document(raw)
    steps, config, provenance = _validate_common(document)

    mind = UnifiedMind(config)
    initial = document.get("initial_mechanisms")
    if initial is not None:
        mind.cognition.restore_mechanism_document(initial)
        mind._initial_mechanisms = copy.deepcopy(initial)
        mind.evolution.adopt_loaded()

    mind.run(steps)
    compatibility = _validate_state_fingerprint(mind, document)
    mind._replay_metadata = {
        "format": document["format"],
        "schema_version": document["schema_version"],
        "provenance": copy.deepcopy(provenance),
        "migration_history": copy.deepcopy(document.get("migration_history", [])),
        "compatibility": compatibility,
        "current_source_sha256": source_fingerprint(),
        "current_numpy": np.__version__,
        "current_scipy": scipy.__version__,
    }
    return mind


def upgrade_replay(path, destination=None):
    """Validate any supported replay and materialize a current v3 checkpoint."""
    source = Path(path)
    target = Path(destination) if destination is not None else source
    mind = load_replay(source)
    save_replay(mind, target)
    return target
