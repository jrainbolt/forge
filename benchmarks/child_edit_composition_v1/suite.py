"""Frozen A66 child composition corpus and settings."""

SUITE = "child-edit-composition-v1"
VERSION = 1
SEED = 42
TEMPERATURE = 0.0
CONTEXT_TOKENS = 8192
OUTPUT_TOKENS = 512
MULTI_TASK_IDS = ("F05", "F06", "F07", "F08")
SINGLE_CONTROL_IDS = ("F01", "F02")
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")
