from __future__ import annotations

import logging

import pytest

from forge.logging import DEFAULT_LOG_FORMAT, configure_logging


@pytest.mark.parametrize(
    ("verbose", "expected_level"),
    [(False, logging.INFO), (True, logging.DEBUG)],
)
def test_logging_uses_expected_level(
    monkeypatch: pytest.MonkeyPatch, verbose: bool, expected_level: int
) -> None:
    calls: list[dict[str, object]] = []

    def fake_basic_config(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    configure_logging(verbose=verbose)

    assert calls == [{"level": expected_level, "format": DEFAULT_LOG_FORMAT}]
