#!/usr/bin/env python3
"""Continue plain text with Meridian Text; this is a base model, not chat.

Copyright 2026 Yifan Ouyang. Licensed under Apache-2.0.
"""
import argparse
from runtime import add_runtime_arguments, load_completion, validate_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    try:
        validate_settings(args)
        if not args.prompt.strip():
            raise ValueError('--prompt must contain text')
        complete = load_completion(args)
        print(complete(args.prompt))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
