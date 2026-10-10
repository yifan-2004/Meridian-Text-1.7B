#!/usr/bin/env python3
"""JSONL completion with validated IDs and resumable, flushed results.

Copyright 2026 Yifan Ouyang. Licensed under Apache-2.0.
"""
import argparse
import json
from pathlib import Path
from runtime import add_runtime_arguments, load_completion, validate_settings


def read_records(path, output=False):
    records = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected an object')
                identity = row.get('id')
                if not isinstance(identity, str) or not identity.strip():
                    raise ValueError('id must be a nonempty string')
                if identity in records:
                    raise ValueError(f'duplicate id: {identity!r}')
                if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                    raise ValueError('prompt must contain text')
                if output and (not isinstance(row.get('completion'), str) or not isinstance(row.get('run'), dict)):
                    raise ValueError('output must include completion and run metadata')
                records[identity] = row
            except (ValueError, TypeError) as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
    return records


def run_metadata(args):
    return {key: getattr(args, key) for key in ('model', 'revision', 'device', 'dtype', 'max_new_tokens')}


def run_batch(args, loader=load_completion):
    validate_settings(args)
    if args.input.resolve() == args.output.resolve():
        raise ValueError('input and output must be different files')
    records = read_records(args.input)
    if not records:
        raise ValueError('input contains no records')
    completed = {}
    if args.output.exists():
        if not args.resume:
            raise ValueError('output exists; use --resume or choose another path')
        completed = read_records(args.output, output=True)
    metadata = run_metadata(args)
    for identity, row in completed.items():
        if identity not in records or row['prompt'] != records[identity]['prompt'] or row['run'] != metadata:
            raise ValueError(f'resume mismatch for {identity!r}; keep the same input and generation settings')
    pending = [row for identity, row in records.items() if identity not in completed]
    if not pending:
        return 0
    # Load before creating output so download/loading failures do not leave empty artifacts.
    complete = loader(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse concurrent or accidentally replacing output when not resuming.
    with args.output.open('a' if args.resume else 'x', encoding='utf-8') as handle:
        # An existing JSONL file may omit its final newline; append a separator first.
        if args.resume and args.output.stat().st_size:
            with args.output.open('rb') as existing:
                existing.seek(-1, 2)
                if existing.read(1) != b'\n':
                    handle.write('\n')
        for row in pending:
            result = {'id': row['id'], 'prompt': row['prompt'], 'completion': complete(row['prompt']), 'run': metadata}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        print(f'Completed {run_batch(args)} new records.')
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
