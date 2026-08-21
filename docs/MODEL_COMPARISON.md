# A3 Model Smoke Comparison

These are observations from one A3 smoke request on an Apple M4 Pro with 48 GB
unified memory. They are not a rigorous benchmark and do not establish a
permanent preferred model. Forge's coding evaluation harness is planned for A8.

Both models received the same generic `ModelRequest`, with an 8,192-token
context, temperature 0, seed 42, and a 192-token output limit. Each was closed
before the other was constructed.

| Profile | Model | Quantization | GGUF size | Construction | Generation | Speed | Metal |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `qwen-small` | Qwen2.5-Coder-7B-Instruct | Q4_K_M | 4,683,073,536 bytes (4.36 GiB) | 0.61 s | 7.22 s | 27.29 tokens/s | 29/29 layers |
| `qwen-large` | Qwen3-Coder-30B-A3B-Instruct | Q4_K_M | 18,632,186,176 bytes (17.35 GiB) | 6.47 s | 4.89 s | 45.11 tokens/s | all 49 layers |

The larger artifact is a mixture-of-experts model with 30.53 billion total
parameters and about 3 billion active parameters. Its faster token generation
on this single short request is an observation, not a general quality or speed
conclusion. It took materially longer to construct and occupied a much larger
Metal-mapped model buffer.

The prompt was:

> Write a C function that returns the maximum value in an array of signed integers.

Both responses produced a direct linear scan initialized from the first array
element. The 7B response supplied a complete function plus an example program,
but did not address an empty array. The 30B-A3B response explicitly handled a
non-positive size using `INT_MIN` and began an example program. Both reached the
deliberately short output limit while adding example code; the requested
function itself was complete and meaningful in both responses.

Runtime logs confirmed the Apple M4 Pro Metal device, unified memory, and full
requested layer offload for both profiles. The 8,192-token contexts consumed
448 MiB of Metal KV cache for the 7B model and 384 MiB for the 30B-A3B model.
