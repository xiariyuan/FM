# Copyright (c) Ruopeng Gao. All Rights Reserved.

import argparse
from utils.misc import yaml_to_dict


def str_to_bool(v):
    """Convert string to boolean value."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    return v  # Return original value if not a boolean string


def update_config_with_kv(config: dict, k: str, v) -> [bool, dict]:
    """
    Update config with a pair of K and V from options.

    Args:
        config: Current config.
        k: A key from options.
        v: A value from options.

    Returns:
        [New config dict, Hit or Not]
    """
    hit = False
    for config_k in config.keys():
        if isinstance(config[config_k], dict):
            hit, config[config_k] = update_config_with_kv(config=config[config_k], k=k, v=v)
            if hit:
                break
        elif config_k == k.upper():
            config[config_k] = str_to_bool(v)
            hit = True
            break
    return hit, config


def update_config(config: dict, option: argparse.Namespace) -> dict:
    """
    Update current config with an option parser.

    Args:
        config: Current config.
        option: Option parser.

    Returns:
        New config dict.
    """
    # v2.0 DO NOT need to check uniqueness
    # if is_unique(config)[0] is False:
    #     raise RuntimeError("Config's key is not unique, Please check the config file.")

    for option_k, option_v in vars(option).items():
        if option_k != "config_path" and option_v is not None:  # except --config-path
            # v2.0 remove hierarchical config setting, using plain config setting.
            # hit, config = update_config_with_kv(config=config, k=option_k, v=option_v)
            config_k = option_k.upper()
            if config_k in config:
                config[config_k] = str_to_bool(option_v)
            else:
                raise RuntimeError(f"The option '{option_k}' is not appeared in .yaml config file.")
    return config


def is_unique(config: dict, keys_set: set = None) -> [bool, set]:
    """
    Check whether the keys in config are unique.

    Args:
        config: Config dict.
        keys_set: Current keys set.

    Returns:
        [Whether the keys are unique, Current keys set]
    """
    if keys_set is None:
        keys_set = set()

    for k in config.keys():
        if k in keys_set:
            return False, keys_set
        else:
            keys_set.add(k)
        if isinstance(config[k], dict):
            hit, keys_set = is_unique(config[k], keys_set=keys_set)
            if hit is False:
                return False, keys_set

    return True, keys_set


def load_super_config(config: dict, super_config_path: str | None):
    if super_config_path is None:
        return config
    else:
        super_config = yaml_to_dict(super_config_path)
        super_config = load_super_config(super_config, super_config.get("SUPER_CONFIG_PATH", None))
        super_config.update(config)
        return super_config
