from collections.abc import Mapping

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def parse_enabled(settings: Mapping[str, str], key: str, *, default: bool) -> bool:
    value = settings.get(key)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean setting: {key}")


def parse_timeout(settings: Mapping[str, str], key: str, default: int) -> int:
    value = int(settings.get(key, default))
    if value <= 0:
        raise ValueError("timeout must be positive")
    return value
