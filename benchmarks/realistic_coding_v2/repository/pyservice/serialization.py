"""Small binary-safe response encoder."""


def encode_response(status: int, body: bytes) -> bytes:
    """Frame a three-digit status and decimal body length before raw bytes."""
    if not 100 <= status <= 599:
        raise ValueError("invalid HTTP status")
    return f"{status}:{len(body)}:".encode("ascii") + body
