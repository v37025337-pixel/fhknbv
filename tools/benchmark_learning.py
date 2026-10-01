"""Measure frozen forecasts on shared experience, then separate policy outcomes.

Seeds 0, 1, 2 are development runs. Seeds 10..15 are the declared evaluation
suite, not used to select forgetting rates or admission rules. This is one
synthetic world family; there is no claim of general intelligence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digital_mind_core import __version__
from digital_mind_core.components import RichWorld
from digital_mind_core.kernel import AdaptivePredictiveModel, KernelConfig, L0PredictiveModel, UnifiedMind
from digital_mind_core.persistence import source_fingerprint


def shared_experience(seed, steps):
    world = RichWorld(seed=seed, regime_change=steps // 2)
    action_rng = np.random.default_rng(seed + 7000)
    baseline = L0PredictiveModel(18, 6, seed + 1, enabled=False)
    adaptive = AdaptivePredictiveModel(18, 6, seed + 1, enabled=False)
    obs = np.zeros(18)
    losses = {'baseline': [], 'adaptive': []}
    for _ in range(steps):
        # Neither learner controls or sees the random action generator's state.
        action = np.clip(action_rng.normal(scale=.6, size=6), -1.2, 1.2)
        frozen = {'baseline': baseline.predict(obs, action), 'adaptive': adaptive.predict(obs, action)}
        following = world.step(action).obs
        for name, model in (('baseline', baseline), ('adaptive', adaptive)):
            prediction, _, _ = model.update(obs, action, following)
            if not np.allclose(prediction, frozen[name], rtol=1e-12, atol=1e-12):
                raise AssertionError('training changed the forecast being evaluated')
            losses[name].append(float(np.mean((following - frozen[name]) ** 2)))
        obs = following
    stages = {'all': slice(None), 'before_change': slice(0, steps // 2),
              'after_change': slice(steps // 2, None), 'final80': slice(steps - 80, None)}
    metrics = {stage: {name: float(np.mean(values[part])) for name, values in losses.items()}
               for stage, part in stages.items()}
    for row in metrics.values():
        row['relative_mse_reduction'] = 1. - row['adaptive'] / row['baseline']
    return {'seed': seed, 'metrics': metrics, 'learning': adaptive.learning_report()}


def closed_loop(seed, steps):
    rows = {}
    for enabled in (False, True):
        mind = UnifiedMind(KernelConfig(seed=seed, adaptive_learning=enabled))
        mind.world.regime_change = steps // 2
        mind.run(steps)
        report = mind.report()
        rows['adaptive' if enabled else 'baseline'] = {
            'prediction': report['prediction'], 'reward_components_mean': report['reward_components_mean'],
            'learning': report['adaptive_learning']}
    return {'seed': seed, 'modes': rows}


def summarize(rows):
    result = {}
    for stage in ('all', 'before_change', 'after_change', 'final80'):
        baseline = float(np.mean([row['metrics'][stage]['baseline'] for row in rows]))
        adaptive = float(np.mean([row['metrics'][stage]['adaptive'] for row in rows]))
        result[stage] = {'baseline_mse': baseline, 'adaptive_mse': adaptive,
                        'relative_mse_reduction': 1. - adaptive / baseline,
                        'improved_seeds': sum(row['metrics'][stage]['adaptive'] <
                                              row['metrics'][stage]['baseline'] for row in rows),
                        'runs': len(rows)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=640)
    parser.add_argument('--report', type=Path, default=Path('examples/learning_benchmark.json'))
    parser.add_argument('--closed-loop', action='store_true')
    args = parser.parse_args()
    if args.steps < 256:
        parser.error('--steps must be at least 256')
    groups = {'development': [0, 1, 2], 'evaluation': list(range(10, 16))}
    result = {'version': __version__, 'source_sha256': source_fingerprint(), 'steps_per_run': args.steps,
              'protocol': {'shared_experience': 'frozen forecasts on identical random-action transitions',
                'l0_innovation_in_shared_test': False, 'world_regime_change': args.steps // 2,
                'closed_loop': 'different visited states; rewards reported separately, no paired MSE claim',
                'scope': 'one synthetic world family; declared evaluation seeds; no model retuning'},
              'groups': {}}
    for group, seeds in groups.items():
        rows = []
        policies = []
        for seed in seeds:
            print(f'{group}: seed={seed}, steps={args.steps}', file=sys.stderr, flush=True)
            rows.append(shared_experience(seed, args.steps))
            if args.closed_loop:
                policies.append(closed_loop(seed, args.steps))
        result['groups'][group] = {'seeds': seeds, 'summary': summarize(rows),
                                   'shared_experience': rows, 'closed_loop': policies}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: value['summary'] for name, value in result['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
