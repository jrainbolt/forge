from __future__ import annotations

from forge.retrieval import (
    SEMANTIC_WEIGHT,
    RetrievalCandidate,
    SourceKind,
    classify_source,
    rank_candidates,
    tokenize,
)


def _candidate(
    path: str,
    similarity: float,
    *,
    symbol: str | None = None,
    text: str = "",
    start: int = 1,
    digest: str | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        path,
        start,
        start + 4,
        "python",
        symbol,
        symbol,
        "function" if symbol else "module",
        digest or f"{path}:{start}",
        similarity,
        text,
    )


def test_source_classification_is_centralized_and_general() -> None:
    assert classify_source("src/forge/agent_loop.py") is SourceKind.IMPLEMENTATION
    assert classify_source("tests/test_agent_loop.py") is SourceKind.TEST
    assert classify_source("docs/ARCHITECTURE.md") is SourceKind.DOCUMENTATION
    assert classify_source("pyproject.toml") is SourceKind.CONFIGURATION
    assert (
        classify_source("src/package.egg-info/PKG-INFO")
        is SourceKind.GENERATED_METADATA
    )
    assert classify_source("notes.txt") is SourceKind.OTHER_TEXT


def test_tokenizer_splits_paths_snake_case_camel_case_and_dotted_names() -> None:
    assert tokenize("ContextPlanner.compact_observations src/forge/context.py") >= {
        "context",
        "planner",
        "compact",
        "observations",
        "src",
        "forge",
    }


def test_lexical_and_structural_evidence_promote_matching_implementation() -> None:
    docs = _candidate("docs/ARCHITECTURE.md", 0.80, text="overview")
    code = _candidate(
        "src/forge/context_planner.py",
        0.78,
        symbol="ContextPlanner.compact_observations",
        text="discard old observations from the context budget",
    )
    ranked = rank_candidates(
        "where does context compaction discard observations", (docs, code), limit=2
    )
    assert ranked[0].path == code.path
    assert ranked[0].lexical_score > ranked[1].lexical_score
    assert ranked[0].source_kind is SourceKind.IMPLEMENTATION


def test_semantic_signal_remains_dominant() -> None:
    best_semantic = _candidate("docs/guide.md", 1.0)
    lexical_code = _candidate(
        "src/answer.py", 0.5, symbol="approval_policy", text="approval policy"
    )
    ranked = rank_candidates("approval policy", (lexical_code, best_semantic), limit=2)
    assert ranked[0] is not lexical_code
    assert SEMANTIC_WEIGHT > 0.8


def test_ties_are_deterministic_and_exact_candidates_are_deduplicated() -> None:
    candidates = (
        _candidate("z.py", 0.7, digest="same"),
        _candidate("a.py", 0.7, digest="same"),
        _candidate("b.py", 0.7),
    )
    ranked = rank_candidates("unmatched", candidates, limit=3)
    assert [item.path for item in ranked] == ["a.py", "b.py"]


def test_result_diversity_caps_one_file_when_alternatives_exist() -> None:
    crowded = tuple(
        _candidate("src/crowded.py", 0.9 - index / 100, start=index * 10 + 1)
        for index in range(5)
    )
    alternatives = (
        _candidate("src/other.py", 0.70),
        _candidate("src/third.py", 0.69),
    )
    ranked = rank_candidates("query", crowded + alternatives, limit=5)
    assert sum(item.path == "src/crowded.py" for item in ranked) == 3
    assert {item.path for item in ranked} == {
        "src/crowded.py",
        "src/other.py",
        "src/third.py",
    }
