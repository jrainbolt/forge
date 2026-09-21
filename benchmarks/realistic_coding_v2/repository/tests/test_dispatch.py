from pyservice.dispatch import dispatch

assert dispatch(" READ ") == "reader"
assert dispatch("write") == "writer"
assert dispatch("DELETE") == "remover"

for action in (None, " ", "unknown"):
    try:
        dispatch(action)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action should be rejected")
