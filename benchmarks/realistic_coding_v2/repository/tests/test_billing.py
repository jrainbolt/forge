from pyservice.billing import invoice_line

assert invoice_line(0) == "TOTAL $0.00"
assert invoice_line(105) == "TOTAL $1.05"
assert invoice_line(12345) == "TOTAL $123.45"

try:
    invoice_line(-1)
except ValueError:
    pass
else:
    raise AssertionError("negative cents should be rejected")
