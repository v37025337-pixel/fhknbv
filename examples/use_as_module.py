"""Use the actual kernel as a computational module, without an external model."""
import argparse
import json
from pathlib import Path
import sys

# Also runs directly from the source checkout before installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digital_mind_core.tool_module import MindModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    module = MindModule()
    simulation = module.execute({'op': 'simulate', 'steps': 96})
    logic = module.execute({'op': 'reason', 'task': {
        'facts': [['error', 'world_prediction']],
        'rules': [[[['error', '$model']], ['needs_review', '$model']]],
        'queries': [['needs_review', 'world_prediction']]}})
    mechanism = module.execute({'op': 'run_mechanism', 'name': 'choose_plan_cost',
                                'inputs': {'history_cost': 2., 'current_cost': 9.}})
    assert logic['inferences'][0]['value']
    assert mechanism['value'] == 2.
    assert mechanism['execution'] == 'L0-Control-IR'
    document = {'version': simulation['version'], 'simulation_steps': simulation['steps'],
                'cognitive_cycles': simulation['cognition']['cycles'],
                'evolution': simulation['mechanism_evolution'],
                'logic_query': logic['inferences'][0], 'mechanism_call': mechanism}
    output = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
