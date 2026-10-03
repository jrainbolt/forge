"""Four additional frozen, coordinated Foundation tasks for A68."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.real_repository_pilot_v1.suite import (
    BUILD,
    CONFIGURE,
    TEST,
    VERIFICATION,
)
from forge.evaluation.realworld import RealWorldLevel, RealWorldTask, SetupReplacement
from forge.interaction import AutonomyMode

ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / "oracle.py"
TASK_IDS = ("F09", "F10", "F11", "F12")
ALL_TASK_IDS = ("F05", "F06", "F07", "F08", *TASK_IDS)
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")
SEED = 42


@dataclass(frozen=True, slots=True)
class GeneralizationTask:
    task_id: str
    production_task: RealWorldTask
    edit_paths: tuple[str, ...]


def _task(
    task_id: str,
    prompt: str,
    paths: tuple[str, str],
    setup: tuple[SetupReplacement, SetupReplacement],
) -> GeneralizationTask:
    production = RealWorldTask(
        task_id,
        RealWorldLevel.BOUNDED_REPAIR,
        AutonomyMode.REPAIR,
        prompt,
        paths,
        allowed_paths=paths,
        expected_changed_paths=paths,
        required_candidate_paths=paths,
        setup=setup,
        configure_command=CONFIGURE,
        build_command=BUILD,
        test_command=TEST,
        verification_plan=VERIFICATION,
        oracle_commands=(("python3", str(ORACLE), task_id),),
        seeds=(SEED,),
        max_mutations=2,
    )
    return GeneralizationTask(task_id, production, paths)


def tasks() -> tuple[GeneralizationTask, ...]:
    return (
        _task(
            "F09",
            "Repair steam recipe validation in src/steam_engine.c and "
            "src/steam_turbine.c so each accepts the documented steam input fluid.",
            ("src/steam_engine.c", "src/steam_turbine.c"),
            (
                SetupReplacement(
                    "src/steam_engine.c",
                    "recipe->input_fluid == FACTORY_FLUID_STEAM",
                    "recipe->input_fluid != FACTORY_FLUID_STEAM",
                ),
                SetupReplacement(
                    "src/steam_turbine.c",
                    "d->input_fluid == FACTORY_FLUID_STEAM",
                    "d->input_fluid != FACTORY_FLUID_STEAM",
                ),
            ),
        ),
        _task(
            "F10",
            "Repair the invalid recipe lookup calls in src/assembler.c and "
            "src/refinery.c so both components build and resolve configured recipes.",
            ("src/assembler.c", "src/refinery.c"),
            (
                SetupReplacement(
                    "src/assembler.c",
                    "factory_assembler_recipe_find(assembler->recipe_id)",
                    "factory_assembler_recipe_find_broken(assembler->recipe_id)",
                ),
                SetupReplacement(
                    "src/refinery.c",
                    "factory_recipe_get(refinery->recipe_id)",
                    "factory_recipe_get_broken(refinery->recipe_id)",
                ),
            ),
        ),
        _task(
            "F11",
            "Repair extractor cycle completion in src/extractor.c and burner release "
            "bounds in src/burner.c so production uses the documented ticks.",
            ("src/extractor.c", "src/burner.c"),
            (
                SetupReplacement(
                    "src/extractor.c",
                    "== FACTORY_EXTRACTOR_PRODUCTION_TICKS)",
                    "!= FACTORY_EXTRACTOR_PRODUCTION_TICKS)",
                ),
                SetupReplacement(
                    "src/burner.c",
                    "elapsed_ticks >= definition->burn_duration_ticks",
                    "elapsed_ticks < definition->burn_duration_ticks",
                ),
            ),
        ),
        _task(
            "F12",
            "Repair definition validation in src/fluid.c and src/fluid_machine.c so "
            "exhaust steam and the documented water-boiling conversion remain valid.",
            ("src/fluid.c", "src/fluid_machine.c"),
            (
                SetupReplacement(
                    "src/fluid.c",
                    "definition->fluid_type <= FACTORY_FLUID_EXHAUST_STEAM",
                    "definition->fluid_type < FACTORY_FLUID_EXHAUST_STEAM",
                ),
                SetupReplacement(
                    "src/fluid_machine.c",
                    "r->recipe_id == FACTORY_FLUID_RECIPE_BOIL_WATER",
                    "r->recipe_id != FACTORY_FLUID_RECIPE_BOIL_WATER",
                ),
            ),
        ),
    )
