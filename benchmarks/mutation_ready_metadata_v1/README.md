# Mutation-ready metadata V1

This evaluator records the proposal boundary using only relative paths, hashes,
IDs, ranges, counts, generations, booleans, and enums. It never records source or
replacement text.

State transitions are strict and one-way:

1. `MECHANICALLY_MATERIALIZABLE` requires a bounded, unique operation set whose
   normalized count and canonical order match its authorized path set.
2. `PRODUCTION_VALIDATABLE` additionally requires matching representation,
   workspace/candidate/child generations, candidate observation identity,
   trusted source hash, authorized range, authority provenance, and group binding.
3. `TRANSACTION_READY` additionally requires preview eligibility and an explicit
   `ready` transaction state.

Missing fields fail closed as `MUTATION_READY_METADATA_INCOMPLETE`. Historical
results without this schema remain readable and are classified
`LEGACY_MUTATION_READY_METADATA_INCOMPLETE`; they are never rewritten or upgraded.
