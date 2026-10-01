import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from .kernel import KernelConfig, UnifiedMind, planning_benchmark
from .checkpoint import load_checkpoint_or_replay, save_checkpoint


def output(document, report):
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    if report:
        Path(report).write_text(text + '\n', encoding='utf-8')
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Unified DIGITAL_MIND + L0 + v0.28 cognitive kernel')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='run a closed-loop simulated agent')
    run.add_argument('--steps', type=int, default=640)
    run.add_argument('--seed', type=int)
    run.add_argument('--horizon', type=int)
    run.add_argument('--no-l0-learning', action='store_true')
    run.add_argument('--no-questions', action='store_true')
    run.add_argument('--no-cognition', action='store_true')
    run.add_argument('--no-evolution', action='store_true')
    run.add_argument('--mechanisms', type=Path, help='load stored mechanisms before starting')
    run.add_argument('--save-mechanisms', type=Path, help='persist generated mechanisms as JSON AST')
    run.add_argument('--report', type=Path)
    run.add_argument('--checkpoint', type=Path)
    run.add_argument('--resume', type=Path, help='restore by deterministic replay, then run --steps more')
    compare = commands.add_parser('compare', help='compare full kernel and three component ablations')
    compare.add_argument('--steps', type=int, default=640)
    compare.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    compare.add_argument('--report', type=Path)
    benchmark = commands.add_parser('benchmark', help='test the shared planner on delayed rewards')
    benchmark.add_argument('--steps', type=int, default=16)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.command == 'benchmark':
        output(planning_benchmark(args.steps), None)
        return
    if args.command == 'run':
        if args.resume:
            if (args.seed is not None or args.horizon is not None or args.no_l0_learning
                    or args.no_questions or args.no_cognition or args.no_evolution or args.mechanisms):
                parser.error('--resume uses its saved config; do not combine with config overrides')
            mind = load_checkpoint_or_replay(args.resume)
        else:
            mind = UnifiedMind(KernelConfig(seed=args.seed if args.seed is not None else 0,
                horizon=args.horizon if args.horizon is not None else 7,
                l0_learning=not args.no_l0_learning, questions=not args.no_questions,
                cognition=not args.no_cognition, mechanism_evolution=not args.no_evolution))
            if args.mechanisms:
                mind.load_mechanisms(args.mechanisms)
        mind.run(args.steps)
        if args.checkpoint:
            save_checkpoint(mind, args.checkpoint)
        if args.save_mechanisms:
            mind.save_mechanisms(args.save_mechanisms)
        output(mind.report(), args.report)
        return
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be unique')
    rows = []
    for seed in args.seeds:
        base = KernelConfig(seed=seed)
        settings = {
            'full': base, 'no_evolution': replace(base, mechanism_evolution=False),
            'no_cognition': replace(base, cognition=False),
            'v01_behavior': replace(base, cognition=False, mechanism_evolution=False),
        }
        for name, config in settings.items():
            print(f'Running {name}, seed={seed}, steps={args.steps}', file=sys.stderr, flush=True)
            mind = UnifiedMind(config)
            mind.run(args.steps)
            report = mind.report()
            rows.append({'mode': name, 'seed': seed, 'prediction': report['prediction'],
                'reward_components_mean': report['reward_components_mean'],
                'questions': report['questions']['created'],
                'explained': report['questions']['predictively_explained'],
                'supported_interventions': sum(e['status'] == 'supported' for e in report['interventions']),
                'goals': report['goals'],
                'l0_weight': report['l0']['mean_innovation_weight'],
                'mechanism_evolution': report['mechanism_evolution'],
                'cognitive_cycles': report['cognition']['cycles']})
    summaries = {}
    for name in settings:
        selected = [r for r in rows if r['mode'] == name]
        summaries[name] = {
            'runs': len(selected),
            'mean_loss': float(np.mean([r['prediction']['mean_loss'] for r in selected])),
            'reward_components_mean': np.mean([r['reward_components_mean'] for r in selected], axis=0).tolist(),
            'mean_questions': float(np.mean([r['questions'] for r in selected])),
        }
    output({'steps_per_run': args.steps, 'seeds': args.seeds, 'summary': summaries,
            'runs': rows, 'shared_planner_benchmark': planning_benchmark()}, args.report)


if __name__ == '__main__':
    main()
