"""Versioned JSON replay checkpoints; no pickle or executable payloads.

Restoration deterministically replays the local simulated experience. It is
linear in the number of steps, rather than a direct learner-state snapshot.
"""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from .kernel import KernelConfig, UnifiedMind


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob('*.py'), *root.rglob('*.l0')]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def state_fingerprint(mind):
    digest = hashlib.sha256()
    for array in (mind.obs, mind.model.W, mind.model.W_fast, mind.self_trace):
        digest.update(np.asarray(array, dtype='<f8').tobytes())
    if hasattr(mind.model, 'learning_state_fingerprint'):
        digest.update(mind.model.learning_state_fingerprint().encode())
    digest.update(json.dumps(mind.report(), sort_keys=True, allow_nan=False).encode())
    digest.update(json.dumps({'cognitive_state': mind.cognition.state,
                              'self_stats': mind.cognition.self_model._stats,
                              'mechanisms': mind.cognition.mechanism_document(),
                              'evolution': mind.evolution.state_document()},
                             sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def save_replay(mind, path):
    if mind._external_tasks:
        raise ValueError('replay covers simulated experience; save external-task mechanisms separately')
    document = {
        'format': 'digital-mind-replay-v2', 'config': asdict(mind.config),
        'steps': mind.fast_steps, 'source_sha256': source_fingerprint(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'state_sha256': state_fingerprint(mind),
        'initial_mechanisms': copy.deepcopy(mind._initial_mechanisms),
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(destination)


def load_replay(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('format') != 'digital-mind-replay-v2':
        raise ValueError('unsupported replay format')
    if document.get('source_sha256') != source_fingerprint():
        raise ValueError('replay source fingerprint differs from the current kernel')
    if document.get('numpy') != np.__version__ or document.get('scipy') != scipy.__version__:
        raise ValueError('replay numerical library versions differ')
    steps = document.get('steps')
    if type(steps) is not int or steps < 0:
        raise ValueError('replay steps must be a nonnegative integer')
    config = document.get('config')
    if not isinstance(config, dict):
        raise ValueError('replay config must be an object')
    try:
        config = KernelConfig(**config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid replay config: {exc}') from exc
    mind = UnifiedMind(config)
    initial = document.get('initial_mechanisms')
    if initial is not None:
        mind.cognition.restore_mechanism_document(initial)
        mind._initial_mechanisms = copy.deepcopy(initial)
        mind.evolution.adopt_loaded()
    mind.run(steps)
    if state_fingerprint(mind) != document.get('state_sha256'):
        raise ValueError('replayed state does not match the saved checkpoint')
    return mind
