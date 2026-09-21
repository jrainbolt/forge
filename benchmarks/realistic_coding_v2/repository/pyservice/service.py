from pyservice.models import Request, Response


def handle(request: Request) -> Response:
    if not request.request_id:
        return Response(400, b"missing request id")
    if not request.payload:
        return Response(204, b"")
    return Response(200, request.payload)
