"""Versioned realworld-v1 task definitions for the Foundation C benchmark."""

from forge.evaluation.realworld import (
    DEFAULT_SEEDS,
    RealWorldLevel,
    RealWorldTask,
    SetupReplacement,
)
from forge.interaction import AutonomyMode

_CONFIGURE = (("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON"),)
_BUILD = ("cmake", "--build", "build")
_TEST = ("ctest", "--test-dir", "build", "--output-on-failure")
_ORACLE = (_BUILD, _TEST)
_CAN_ADVANCE = "return clock != NULL && clock->tick != UINT64_MAX;"
_BROKEN_ADVANCE = "return clock != NULL && clock->tick == UINT64_MAX;"


def foundation_realworld_tasks() -> tuple[RealWorldTask, ...]:
    """Return immutable natural-language tasks without any machine-local paths."""
    return (
        RealWorldTask(
            "E01",
            RealWorldLevel.REPOSITORY_REASONING,
            AutonomyMode.READ,
            "Where does this engine decide whether a submitted command is "
            "structurally valid before the command is applied?",
            ("src/command.c",),
            seeds=DEFAULT_SEEDS,
        ),
        RealWorldTask(
            "E02",
            RealWorldLevel.REPOSITORY_REASONING,
            AutonomyMode.READ,
            "Explain how inserters and logistics endpoints cooperate to move "
            "items between entities, citing the implementation areas involved.",
            ("src/inserter.c", "src/logistics_endpoint.c"),
            seeds=DEFAULT_SEEDS,
        ),
        RealWorldTask(
            "E03",
            RealWorldLevel.REPOSITORY_REASONING,
            AutonomyMode.READ,
            "The simulation clock no longer advances from ordinary tick values. "
            "Diagnose the exact faulty condition and explain what should change.",
            ("src/clock.c",),
            setup=(SetupReplacement("src/clock.c", _CAN_ADVANCE, _BROKEN_ADVANCE),),
            seeds=DEFAULT_SEEDS,
        ),
        RealWorldTask(
            "E04",
            RealWorldLevel.SINGLE_CHANGE,
            AutonomyMode.ASSIST,
            "Fix the defect that prevents the simulation clock from advancing "
            "from ordinary tick values, then verify the change.",
            ("src/clock.c",),
            allowed_paths=("src/clock.c",),
            expected_changed_paths=("src/clock.c",),
            setup=(SetupReplacement("src/clock.c", _CAN_ADVANCE, _BROKEN_ADVANCE),),
            setup_commands=_CONFIGURE,
            build_command=_BUILD,
            test_command=_TEST,
            oracle_commands=_ORACLE,
            max_mutations=1,
        ),
        RealWorldTask(
            "E05",
            RealWorldLevel.SINGLE_CHANGE,
            AutonomyMode.ASSIST,
            "Add one focused regression assertion to the solar tests confirming "
            "that intensity is zero exactly at sunset, then run verification.",
            ("tests/test_solar.c", "src/clock.c"),
            allowed_paths=("tests/test_solar.c",),
            expected_changed_paths=("tests/test_solar.c",),
            setup_commands=_CONFIGURE,
            build_command=_BUILD,
            test_command=_TEST,
            oracle_commands=_ORACLE,
            max_mutations=1,
        ),
        RealWorldTask(
            "E06",
            RealWorldLevel.SINGLE_CHANGE,
            AutonomyMode.ASSIST,
            "Correct the solar intensity boundary so values at and after sunset "
            "are rejected, while preserving daytime behavior, and verify it.",
            ("src/clock.c", "tests/test_solar.c"),
            allowed_paths=("src/clock.c",),
            expected_changed_paths=("src/clock.c",),
            setup=(
                SetupReplacement(
                    "src/clock.c",
                    "time_of_day >= FACTORY_CLOCK_SUNSET",
                    "time_of_day > FACTORY_CLOCK_SUNSET",
                ),
            ),
            setup_commands=_CONFIGURE,
            build_command=_BUILD,
            test_command=_TEST,
            oracle_commands=_ORACLE,
            max_mutations=1,
        ),
        RealWorldTask(
            "E07",
            RealWorldLevel.BOUNDED_REPAIR,
            AutonomyMode.REPAIR,
            "Fix the clock advance guard and verify it. If verification exposes "
            "the adjacent sunset boundary defect, diagnose and repair that too.",
            ("src/clock.c", "tests/test_solar.c"),
            allowed_paths=("src/clock.c",),
            expected_changed_paths=("src/clock.c",),
            setup=(
                SetupReplacement("src/clock.c", _CAN_ADVANCE, _BROKEN_ADVANCE),
                SetupReplacement(
                    "src/clock.c",
                    "time_of_day >= FACTORY_CLOCK_SUNSET",
                    "time_of_day > FACTORY_CLOCK_SUNSET",
                ),
            ),
            setup_commands=_CONFIGURE,
            build_command=_BUILD,
            test_command=_TEST,
            oracle_commands=_ORACLE,
            max_mutations=2,
        ),
        RealWorldTask(
            "E08",
            RealWorldLevel.BOUNDED_REPAIR,
            AutonomyMode.AGENT,
            "Add a clock boundary regression test and the corresponding source "
            "fix as one coordinated two-file task, then verify both.",
            ("src/clock.c", "tests/test_solar.c"),
            allowed_paths=("src/clock.c", "tests/test_solar.c"),
            expected_changed_paths=("src/clock.c", "tests/test_solar.c"),
            max_mutations=2,
            unsupported_reason=(
                "CAPABILITY GAP — multi-file atomic coding task unsupported by "
                "the current non-repair mutation ceiling"
            ),
        ),
    )
