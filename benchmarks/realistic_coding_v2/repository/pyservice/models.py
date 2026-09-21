from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    payload: bytes


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
