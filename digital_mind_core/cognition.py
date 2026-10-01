"""Shared symbolic cognition and bounded synthesis, executing new rules via L0.

The uploaded v0.28 branch remains available for compatibility. This facade adds
detached public state, independent admission checks, JSON AST persistence and
L0 execution of every newly synthesized mechanism.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile

from . import l0
from .l0.control import execute_relation


LEGACY_ROOT = Path(__file__).with_name('legacy_v028')


def load_legacy(name, filename):
    key = 'digital_mind_core._legacy_' + name
    if key in sys.modules:
        return sys.modules[key]
    path = LEGACY_ROOT / filename
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


_branch = load_legacy('cognitive', 'v0.18_cognitive_core.py')
genesis = _branch._v028_load_genesis_module()


def expression_document(expr):
    return {'op': expr.op, 'args': [expression_document(x) if hasattr(x, 'op') else x
                                  for x in expr.args]}


def expression_from_document(document, variables):
    count = 0
    arities = {'neg': 1, 'abs': 1, 'add': 2, 'sub': 2, 'mul': 2,
               'min': 2, 'max': 2, 'lt': 2, 'le': 2, 'gt': 2, 'ge': 2, 'eq': 2, 'if': 3}

    def decode(node, depth):
        nonlocal count
        count += 1
        if count > 128 or depth > 24:
            raise ValueError('mechanism expression exceeds its size or depth limit')
        if not isinstance(node, dict) or set(node) != {'op', 'args'}:
            raise ValueError('invalid expression object')
        op, args = node['op'], node['args']
        if not isinstance(op, str) or not isinstance(args, list):
            raise ValueError('invalid expression fields')
        if op in {'var', 'const'}:
            if len(args) != 1:
                raise ValueError('invalid expression arity')
            if op == 'var' and (not isinstance(args[0], str) or args[0] not in variables):
                raise ValueError('undeclared mechanism variable')
            if op == 'const' and not genesis._is_number(args[0]):
                raise ValueError('mechanism constant must be finite and numeric')
            return genesis.Expr(op, tuple(args), 1)
        if op not in arities or len(args) != arities[op]:
            raise ValueError('unsupported mechanism operator or arity')
        children = tuple(decode(x, depth + 1) for x in args)
        return genesis.Expr(op, children, 1 + sum(x.cost for x in children))

    return decode(document, 0)


def source_hash(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def legacy_expression(function):
    """Convert a packaged pure expression to the same AST used by synthesis."""
    tree = ast.parse(inspect.getsource(function))
    definition = tree.body[0]
    args = [x.arg for x in definition.args.args]
    if len(definition.body) != 1 or not isinstance(definition.body[0], ast.Return):
        raise ValueError('legacy mechanism must be a pure return expression')
    binary = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul'}
    comparisons = {ast.Lt: 'lt', ast.LtE: 'le', ast.Gt: 'gt', ast.GtE: 'ge', ast.Eq: 'eq'}

    def convert(node):
        if isinstance(node, ast.Name) and node.id in args:
            return genesis.Expr('var', (node.id,), 1)
        if isinstance(node, ast.Constant) and genesis._is_number(node.value):
            return genesis.Expr('const', (node.value,), 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            op, children = 'neg', (convert(node.operand),)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            op, children = binary[type(node.op)], (convert(node.left), convert(node.right))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in comparisons:
            op = comparisons[type(node.ops[0])]
            children = (convert(node.left), convert(node.comparators[0]))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in {'abs', 'min', 'max'} and not node.keywords):
            op, children = node.func.id, tuple(convert(x) for x in node.args)
        elif isinstance(node, ast.IfExp):
            op, children = 'if', (convert(node.test), convert(node.body), convert(node.orelse))
        else:
            raise ValueError('legacy mechanism contains an unsupported expression')
        return genesis.Expr(op, children, 1 + sum(x.cost for x in children))

    return args, convert(definition.body[0].value)


def atomic_json(path, document):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class CognitiveCore(_branch.CognitiveCore):
    def __init__(self):
        super().__init__()
        self._l0_mechanisms = {}
        self.mechanism_calls = 0
        for name, entry in self._persisted_mechanisms.items():
            args, expression = legacy_expression(entry['function'])
            source = genesis.MechanismGenesis._function_source(name, args, expression)
            result = genesis.SynthesisResult('FOUND', name, args, expression, source, expression.cost,
                                              0, 0, 0, 0, 'migrated packaged v0.28 mechanism')
            self._l0_mechanisms[name] = self._compile_mechanism(result)
            self._mechanism_objects[name] = result
            self.mechanism_registry[name] = {**genesis.result_dict(result),
                'execution': 'L0-Control-IR', 'admission': 'legacy_examples_only',
                'origin': 'v0.28 packaged mechanism', 'source_sha256': source_hash(source),
                'legacy_source_sha256': entry.get('source_sha256')}

    def process(self, task):
        return copy.deepcopy(super().process(copy.deepcopy(task)))

    cycle = process

    def mechanism_info(self, name=None):
        if name is None:
            return copy.deepcopy({**super().mechanism_info(), **self.mechanism_registry})
        return copy.deepcopy(super().mechanism_info(name))

    def _compile_mechanism(self, result):
        module = l0.lower_foreign_control(result.source, 'python', result.name)
        if module.unresolved_count or not all(r.resolved for r in module.relations):
            raise ValueError('generated expression cannot be executed by L0')
        return module

    def synthesize_mechanism(self, spec, *, validation_examples=None, max_rounds=8,
                             holdout_examples=None, engine_options=None):
        name = spec.get('name', 'generated_mechanism')
        previous = (self._mechanism_objects.get(name), self.mechanism_registry.get(name),
                    self._l0_mechanisms.get(name))
        summary = super().synthesize_mechanism(spec, validation_examples=validation_examples,
                                               max_rounds=max_rounds, engine_options=engine_options)
        if summary['status'] != 'FOUND':
            return copy.deepcopy(summary)
        result = self._mechanism_objects[name]
        try:
            compiled = self._compile_mechanism(result)
            # A separate fixed holdout is not fed back into synthesis.
            examples = holdout_examples if holdout_examples is not None else spec['examples']
            if holdout_examples is not None and len(examples) < 2:
                raise ValueError('admission requires at least two held-out examples')
            for item in examples:
                expected = item['output']
                interpreted = genesis.execute_result(result, item['inputs'])
                emitted = execute_relation(compiled, name, [item['inputs'][a] for a in result.args])
                if not genesis._same(interpreted, expected) or interpreted != emitted:
                    raise ValueError('held-out example or L0 semantic equivalence failed')
        except (ValueError, RuntimeError, TypeError, KeyError, OverflowError) as exc:
            for mapping, old in zip((self._mechanism_objects, self.mechanism_registry,
                                     self._l0_mechanisms), previous):
                if old is None:
                    mapping.pop(name, None)
                else:
                    mapping[name] = old
            summary['status'] = 'REJECTED'
            summary['explanation'] = str(exc)
            self.self_model.observe('mechanism_admission', False)
            return copy.deepcopy(summary)
        summary['execution'] = 'L0-Control-IR'
        summary['source_sha256'] = source_hash(result.source)
        summary['admission'] = 'holdout_passed' if holdout_examples is not None else 'examples_only'
        summary['holdout_count'] = len(examples) if holdout_examples is not None else 0
        summary['holdout_examples'] = copy.deepcopy(examples) if holdout_examples is not None else []
        self.mechanism_registry[name] = copy.deepcopy(summary)
        self._l0_mechanisms[name] = compiled
        self.self_model.observe('mechanism_admission', True)
        return copy.deepcopy(summary)

    def run_mechanism(self, name, **inputs):
        if name not in self._l0_mechanisms:
            return super().run_mechanism(name, **inputs)
        result = self._mechanism_objects[name]
        if set(inputs) != set(result.args) or not all(genesis._is_number(v) for v in inputs.values()):
            raise ValueError('mechanism inputs must match its finite numeric arguments')
        value = execute_relation(self._l0_mechanisms[name], name, [inputs[a] for a in result.args])
        if not genesis._is_number(value):
            raise ValueError('mechanism produced a non-finite scalar')
        self.mechanism_calls += 1
        return value

    def mechanism_document(self):
        return {'format': 'digital-mind-mechanisms-v1', 'mechanisms': [
            {'name': name, 'args': result.args, 'expression': expression_document(result.expression),
             'source_sha256': source_hash(result.source),
             'summary': copy.deepcopy(self.mechanism_registry[name])}
            for name, result in sorted(self._mechanism_objects.items())]}

    def save_mechanisms(self, path):
        atomic_json(path, self.mechanism_document())

    def load_mechanisms(self, path):
        if Path(path).stat().st_size > 1_000_000:
            raise ValueError('mechanism store exceeds 1 MB')
        self.restore_mechanism_document(json.loads(Path(path).read_text(encoding='utf-8')))

    def restore_mechanism_document(self, document):
        if not isinstance(document, dict) or document.get('format') != 'digital-mind-mechanisms-v1':
            raise ValueError('unsupported mechanism store format')
        entries = document.get('mechanisms')
        if not isinstance(entries, list) or len(entries) > 64:
            raise ValueError('invalid number of stored mechanisms')
        staged = []
        names = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError('invalid stored mechanism')
            name, args = item.get('name'), item.get('args')
            if not isinstance(name, str) or name in names or not isinstance(args, list):
                raise ValueError('invalid or duplicate mechanism name/arguments')
            if not args or not all(isinstance(a, str) for a in args) or len(set(args)) != len(args):
                raise ValueError('mechanism arguments must be unique names')
            expr = expression_from_document(item.get('expression'), args)
            source = genesis.MechanismGenesis._function_source(name, args, expr)
            if source_hash(source) != item.get('source_sha256'):
                raise ValueError('mechanism source hash mismatch')
            summary = item.get('summary')
            if not isinstance(summary, dict) or summary.get('status') != 'FOUND':
                raise ValueError('stored mechanism is not registered')
            result = genesis.SynthesisResult('FOUND', name, args, expr, source, expr.cost,
                summary.get('examples_passed', 0), summary.get('examples_total', 0),
                0, 0, 'restored from validated JSON expression')
            compiled = self._compile_mechanism(result)
            if summary.get('admission') == 'holdout_passed':
                certificate = summary.get('holdout_examples')
                if not isinstance(certificate, list) or len(certificate) < 2:
                    raise ValueError('stored mechanism lacks its held-out admission examples')
                for example in certificate:
                    interpreted = genesis.execute_result(result, example['inputs'])
                    emitted = execute_relation(compiled, name, [example['inputs'][a] for a in args])
                    if interpreted != emitted or not genesis._same(interpreted, example['output']):
                        raise ValueError('stored mechanism failed revalidation')
            summary = {**summary, 'name': name, 'args': args, 'source': source,
                       'cost': expr.cost, 'source_sha256': source_hash(source),
                       'execution': 'L0-Control-IR'}
            staged.append((name, result, copy.deepcopy(summary), compiled))
            names.add(name)
        # Publish only after every document entry has passed structural checks.
        for name, result, summary, compiled in staged:
            self._mechanism_objects[name] = result
            self.mechanism_registry[name] = summary
            self._l0_mechanisms[name] = compiled

    def connect(self, target, **kwargs):
        if not hasattr(self, '_autoconnect'):
            runtime = load_legacy('autoconnect', 'v0.14_autoconnect_runtime.py')
            self._autoconnect = runtime.AutoConnectRuntime(runtime_kwargs={
                'memory_max_episodes': 100, 'memory_target_episodes': 50,
                'memory_max_concepts': 200})
        return self._autoconnect.connect(target, **kwargs)
