from __future__ import annotations

from typing import Any, Dict, Optional


_PROFILE_KEY_MAP = {
    "EXP_FILE": "BYTETRACK_EXP_FILE",
    "BYTETRACK_EXP_FILE": "BYTETRACK_EXP_FILE",
    "CKPT": "BYTETRACK_CKPT",
    "BYTETRACK_CKPT": "BYTETRACK_CKPT",
    "FP16": "BYTETRACK_FP16",
    "BYTETRACK_FP16": "BYTETRACK_FP16",
    "TEST_SIZE": "BYTETRACK_TEST_SIZE",
    "BYTETRACK_TEST_SIZE": "BYTETRACK_TEST_SIZE",
    "CONF_THRE": "BYTETRACK_CONF_THRE",
    "BYTETRACK_CONF_THRE": "BYTETRACK_CONF_THRE",
    "NMS_THRE": "BYTETRACK_NMS_THRE",
    "BYTETRACK_NMS_THRE": "BYTETRACK_NMS_THRE",
    "CLASS_AGNOSTIC_NMS": "BYTETRACK_CLASS_AGNOSTIC_NMS",
    "BYTETRACK_CLASS_AGNOSTIC_NMS": "BYTETRACK_CLASS_AGNOSTIC_NMS",
    "DET_SOURCE": "BYTETRACK_DET_SOURCE",
    "BYTETRACK_DET_SOURCE": "BYTETRACK_DET_SOURCE",
    "EXTERNAL_DET_ROOT": "EXTERNAL_DET_ROOT",
    "DET_ROOT": "EXTERNAL_DET_ROOT",
    "EXTERNAL_DET_PATTERN": "EXTERNAL_DET_PATTERN",
    "DET_PATTERN": "EXTERNAL_DET_PATTERN",
    "EXTERNAL_DET_FILE_BY_SEQ": "EXTERNAL_DET_FILE_BY_SEQ",
    "DET_FILE_BY_SEQ": "EXTERNAL_DET_FILE_BY_SEQ",
}


def _upper_key_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        out[str(key).upper()] = value
    return out


def _pick_profile_name(
    config: Dict[str, Any],
    dataset_name: Optional[str],
    explicit_profile: Optional[str],
) -> Optional[str]:
    if explicit_profile is not None and str(explicit_profile).strip():
        return str(explicit_profile).strip()

    by_dataset = config.get("BYTETRACK_PROFILE_BY_DATASET", {})
    if not isinstance(by_dataset, dict):
        by_dataset = {}
    by_dataset_upper = _upper_key_dict(by_dataset)

    dataset_candidates = []
    if dataset_name:
        dataset_candidates.extend([
            str(dataset_name),
            str(dataset_name).upper(),
            str(dataset_name).lower(),
        ])
    for candidate in dataset_candidates:
        if candidate in by_dataset:
            return str(by_dataset[candidate]).strip()
        if candidate.upper() in by_dataset_upper:
            return str(by_dataset_upper[candidate.upper()]).strip()

    for fallback_key in ("DEFAULT", "__DEFAULT__"):
        if fallback_key in by_dataset_upper and str(by_dataset_upper[fallback_key]).strip():
            return str(by_dataset_upper[fallback_key]).strip()

    cfg_profile = config.get("BYTETRACK_PROFILE", None)
    if cfg_profile is not None and str(cfg_profile).strip():
        return str(cfg_profile).strip()

    return None


def resolve_bytetrack_profile(
    config: Dict[str, Any],
    dataset_name: Optional[str] = None,
    explicit_profile: Optional[str] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    profiles = config.get("BYTETRACK_PROFILES", {})
    if not isinstance(profiles, dict) or len(profiles) == 0:
        return config, None

    profile_name = _pick_profile_name(
        config=config,
        dataset_name=dataset_name,
        explicit_profile=explicit_profile,
    )
    if profile_name is None:
        return config, None

    chosen_profile = None
    for key, value in profiles.items():
        if str(key) == profile_name:
            chosen_profile = value
            break
    if chosen_profile is None:
        profiles_upper = _upper_key_dict(profiles)
        chosen_profile = profiles_upper.get(profile_name.upper(), None)

    if not isinstance(chosen_profile, dict):
        available = sorted([str(key) for key in profiles.keys()])
        raise KeyError(
            f"Invalid BYTETRACK profile '{profile_name}'. "
            f"Available profiles: {available}"
        )

    for raw_key, value in chosen_profile.items():
        mapped_key = _PROFILE_KEY_MAP.get(str(raw_key).upper(), None)
        if mapped_key is None:
            continue
        config[mapped_key] = value

    config["BYTETRACK_PROFILE_SELECTED"] = profile_name
    return config, profile_name
