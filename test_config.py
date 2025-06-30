#!/usr/bin/env python3
"""Test config handling"""

from omegaconf import OmegaConf

# Test OmegaConf boolean handling
config = OmegaConf.create({'pda': {'enable': True}})
print(f"Original: {config.pda.enable} (type: {type(config.pda.enable)})")

# Test override with string "false"
OmegaConf.update(config, "pda.enable", "false")
print(f"After override with 'false': {config.pda.enable} (type: {type(config.pda.enable)})")

# Test override with boolean False
OmegaConf.update(config, "pda.enable", False)
print(f"After override with False: {config.pda.enable} (type: {type(config.pda.enable)})")

# Convert to container
container = OmegaConf.to_container(config)
print(f"Container: {container['pda']['enable']} (type: {type(container['pda']['enable'])})")