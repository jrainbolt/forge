"""Distribution policy permits only documented installed evaluation resources."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_wheel import (
    REQUIRED,
    RUNTIME_EVAL_RESOURCE_FILES,
    archive_files,
    content_violations,
)


def test_required_installed_evaluation_resource_is_explicitly_allowed() -> None:
    assert len(RUNTIME_EVAL_RESOURCE_FILES) == 9
    assert content_violations(REQUIRED) == ((), ())


@pytest.mark.parametrize(
    "path",
    (
        "forge/evaluation/fixtures/eval_repo/extra.py",
        "forge/evaluation/fixtures/other_repo/src/app.py",
        "forge/other/fixtures/hidden.txt",
        "forge/other/eval_repo/data.py",
        "tests/fixtures/eval_repo/src/tinyqueue/retry.py",
        "benchmarks/private_case/source.py",
        "benchmark_repo/source.py",
        "eval-results/run.json",
        "replay-data/case.json",
        "replay_data/case.json",
        "replay-results/output.json",
        "replays/case.json",
        "generated-tests/test_candidate.py",
        "generated_test/test_candidate.py",
        ".forge-exec/transactions/00.new",
        "models/local.gguf",
        "config/forge.toml",
        "forge/__pycache__/module.pyc",
        "forge/../secret.py",
        "forge/./secret.py",
        "forge\\evaluation\\fixture.py",
    ),
)
def test_unintended_distribution_content_is_rejected(path: str) -> None:
    missing, forbidden = content_violations(REQUIRED | {path})
    assert missing == ()
    assert forbidden == (path,)


def test_missing_installed_runtime_resource_is_rejected() -> None:
    missing, forbidden = content_violations(
        REQUIRED - {"forge/evaluation/fixtures/eval_repo/tests/test_retry.py"}
    )
    assert missing == ("forge/evaluation/fixtures/eval_repo/tests/test_retry.py",)
    assert forbidden == ()


def test_wheel_and_sdist_readers_normalize_same_resource_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("forge/evaluation/fixtures/eval_repo/README.md", "fixture")
    sdist = tmp_path / "fixture.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        content = b"fixture"
        info = tarfile.TarInfo(
            "forge-0.1/src/forge/evaluation/fixtures/eval_repo/README.md"
        )
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    expected = frozenset({"forge/evaluation/fixtures/eval_repo/README.md"})
    assert archive_files(wheel) == expected
    assert archive_files(sdist) == expected
