"""Task state shared by queue components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    payload: str
    attempts: int = 0
