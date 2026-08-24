from forge.evaluation import ROUTING_V1, ROUTING_V1_TASKS, load_routing_suite


def test_routing_v1_is_fixed_six_task_suite() -> None:
    assert load_routing_suite(ROUTING_V1) is ROUTING_V1_TASKS
    assert [task.task_id for task in ROUTING_V1_TASKS] == [
        "R01",
        "R02",
        "R03",
        "R04",
        "R05",
        "R06",
    ]
