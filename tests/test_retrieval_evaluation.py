from forge.evaluation import RETRIEVAL_V1_TASKS, RetrievalTask, evaluate_retrieval
from forge.retrieval import SourceKind
from forge.semantic_index import SemanticMatch


def _match(path: str, kind: SourceKind) -> SemanticMatch:
    return SemanticMatch(path, 1, 5, "python", None, None, "module", 0.8, kind)


def test_retrieval_v1_has_six_fixed_model_independent_tasks() -> None:
    assert [task.task_id for task in RETRIEVAL_V1_TASKS] == [
        "H01",
        "H02",
        "H03",
        "H04",
        "H05",
        "H06",
    ]


def test_retrieval_metrics_compare_raw_and_reranked_positions() -> None:
    docs = _match("docs/ARCHITECTURE.md", SourceKind.DOCUMENTATION)
    generated = _match("pkg.egg-info/PKG-INFO", SourceKind.GENERATED_METADATA)
    code = _match("src/target.py", SourceKind.IMPLEMENTATION)

    class FakeIndex:
        def search_raw(self, query: str, *, limit: int):
            return (generated, docs, code)

        def search(self, query: str, *, limit: int):
            return (code, docs)

    task = RetrievalTask("H00", "target", ("src/target.py",))
    results, metrics = evaluate_retrieval(FakeIndex(), (task,))
    assert results[0].raw_rank == 3
    assert results[0].reranked_rank == 1
    assert metrics.raw_top1 == 0.0
    assert metrics.reranked_top1 == 1.0
    assert metrics.raw_generated_metadata == 1
    assert metrics.reranked_generated_metadata == 0
    assert metrics.raw_docs_before_implementation == 1
    assert metrics.reranked_docs_before_implementation == 0
