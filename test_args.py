#!/usr/bin/env python3
"""Test argument parsing"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yml')
parser.add_argument('--override', nargs='+', default=[])
args = parser.parse_args()

print(f"Config: {args.config}")
print(f"Number of overrides: {len(args.override)}")
print(f"Overrides list: {args.override}")

for i, override in enumerate(args.override):
    print(f"  Override {i}: '{override}'")
    if '=' in override:
        key, value = override.split('=', 1)
        print(f"    Key: '{key}', Value: '{value}'")