"""Offline regression tests: fake completions, never load model weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from batch_complete import read_records, run_batch
from runtime import validate_settings


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = argparse.Namespace(input=root/'in.jsonl', output=root/'out.jsonl', resume=False,
            model='local-test', revision='fixed', device='cpu', dtype='auto', max_new_tokens=20)
        self.args.input.write_text('\n'.join(json.dumps({'id': str(i), 'prompt': f'input {i}'}) for i in range(3)))

    def test_resume_after_interrupted_generation(self):
        def loader(args):
            def complete(prompt):
                if prompt == 'input 1':
                    raise RuntimeError('simulated interruption')
                return 'test output'
            return complete
        with self.assertRaises(RuntimeError):
            run_batch(self.args, loader)
        self.assertEqual(len(read_records(self.args.output, True)), 1)
        self.args.resume = True
        calls = []
        self.assertEqual(run_batch(self.args, lambda _: lambda p: calls.append(p) or 'test'), 2)
        self.assertEqual(calls, ['input 1', 'input 2'])
        self.assertEqual(run_batch(self.args, lambda _: self.fail('must not reload')), 0)

    def test_duplicate_ids_rejected_before_loading(self):
        self.args.input.write_text('{"id":"x","prompt":"a"}\n{"id":"x","prompt":"b"}\n')
        with self.assertRaisesRegex(ValueError, 'duplicate id'):
            run_batch(self.args, lambda _: self.fail('must not load'))
        self.assertFalse(self.args.output.exists())

    def test_output_protected(self):
        self.args.output.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'output exists'):
            run_batch(self.args, lambda _: self.fail('must not load'))
        self.assertEqual(self.args.output.read_text(), 'keep')

    def test_resume_rejects_changed_prompt_or_settings(self):
        run_batch(self.args, lambda _: lambda p: 'test')
        self.args.resume = True
        self.args.max_new_tokens = 21
        with self.assertRaisesRegex(ValueError, 'resume mismatch'):
            run_batch(self.args, lambda _: self.fail('must not load'))
        self.args.max_new_tokens = 20
        self.args.input.write_text(self.args.input.read_text().replace('input 0', 'changed'))
        with self.assertRaisesRegex(ValueError, 'resume mismatch'):
            run_batch(self.args)

    def test_truncated_output_is_not_silently_skipped(self):
        self.args.output.write_text('{"id":')
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, ':1:'):
            run_batch(self.args)

    def test_resume_without_newline(self):
        run_batch(self.args, lambda _: lambda p: 'test')
        first = self.args.output.read_text().splitlines()[0]
        self.args.output.write_text(first)
        self.args.resume = True
        self.assertEqual(run_batch(self.args, lambda _: lambda p: 'test'), 2)
        self.assertEqual(len(read_records(self.args.output, True)), 3)

    def test_id_and_prompt_types(self):
        for row in [{'id': 1, 'prompt': 'text'}, {'id': 'x', 'prompt': ' '}, []]:
            self.args.input.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                run_batch(self.args)

    def test_context_and_dtype_validation(self):
        self.args.max_new_tokens = 4096
        with self.assertRaises(ValueError): validate_settings(self.args)
        self.args.max_new_tokens = 20
        self.args.dtype = 'float16'
        with self.assertRaises(ValueError): validate_settings(self.args)


class EvaluationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Minimal registry stand-in isolates pure postprocessing from OpenCompass.
        class Registry:
            def register_module(self, *args, **kwargs):
                return lambda function: function
        registry = types.ModuleType('opencompass.registry')
        registry.DICT_POSTPROCESSORS = Registry()
        registry.TEXT_POSTPROCESSORS = Registry()
        package = types.ModuleType('opencompass')
        sys.modules['opencompass'] = package
        sys.modules['opencompass.registry'] = registry
        spec = importlib.util.spec_from_file_location('meridian_postprocessors', ROOT/'research/evaluation/genesis_iii_eval/postprocessors.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_empty_answers_cannot_score_correct(self):
        self.assertFalse(self.module.answers_match('', ''))
        result = self.module.judge_answer_extraction_postprocess({'0': {'prediction': '', 'gold': ''}}, '')
        self.assertEqual(result['correct_count'], 0)
        self.assertEqual(result['no_answer_count'], 1)
        self.assertEqual(result['no_answer_rate'], 100)

    def test_empty_closed_reasoning_is_no_answer(self):
        self.assertEqual(self.module.strip_thinking_content('<think>secret reasoning</think>'), 'NO_ANSWER')

    def test_latest_mixed_marker_is_used(self):
        text = '<think>first</think><reasoning>second</reasoning> B'
        self.assertEqual(self.module.strip_thinking_content(text), 'B')

    def test_whitespace_before_unclosed_reasoning(self):
        self.assertEqual(self.module.strip_thinking_content('  <think>unfinished'), 'NO_ANSWER')

    def test_reopened_reasoning_is_not_answer(self):
        self.assertEqual(self.module.strip_thinking_content('<think>x</think><reasoning>unfinished'), 'NO_ANSWER')

    def test_standard_answer_preserved(self):
        self.assertEqual(self.module.strip_thinking_content('<think>work</think>\nB'), 'B')
        self.assertTrue(self.module.answers_match('(B)', 'B'))


if __name__ == '__main__':
    unittest.main()
