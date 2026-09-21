"""Public response framing API."""

from pyservice.models import Response
from pyservice.serialization import encode_response


def wire_response(response: Response) -> bytes:
    return encode_response(response.status, response.body)
