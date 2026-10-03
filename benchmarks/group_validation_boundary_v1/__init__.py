"""A69 evaluator mirror of production grouped-validation prerequisites."""

from benchmarks.group_validation_boundary_v1.protocol import (
    GroupValidationFailure,
    ValidationClass,
    validate_v1,
)

__all__ = ("GroupValidationFailure", "ValidationClass", "validate_v1")
