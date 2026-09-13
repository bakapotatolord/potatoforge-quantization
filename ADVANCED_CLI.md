# Advanced CLI

The commands below use `uv run potatoforge`. In a Python environment installed
with `pip`, replace that prefix with `potatoforge`. Run
`potatoforge --help` for the short command map.

## Validation boundary

Audits and estimates describe tensor storage and reconstruction proxies. A
successful conversion writes an artifact, but does not prove runtime
compatibility, speed, or image quality. Test the output with the intended
ComfyUI runtime and loader before relying on it.

## Inspect headers

Inspect a safetensors header without reading payloads:

```powershell
uv run potatoforge inspect model path\to\model.safetensors
```

Write a JSON report or show quantization metadata:

```powershell
uv run potatoforge inspect model path\to\model.safetensors `
    --output reports\model.json

uv run potatoforge inspect model path\to\quantized.safetensors `
    --quantization
```

Quantized outputs record their summary in `__metadata__` and their layer map
in `potatoforge.quantization_layers`.

Inspect a LoRA checkpoint without loading tensor payloads:

```powershell
uv run potatoforge inspect lora `
    path\to\lora.safetensors `
    --output reports\lora-inventory.json
```

## Merge LoRA files

Merge one or more LoRA files into a new checkpoint while preserving source
dtypes:

```powershell
uv run potatoforge lora merge `
    path\to\source.safetensors `
    path\to\merged.safetensors `
    --adapter-path path\to\style-a.safetensors `
    --adapter-strength 0.65 `
    --adapter-path path\to\style-b.safetensors `
    --adapter-strength 0.40
```

LoRA targets use an exact source-name match first, then one unique
dot-boundary suffix match. Missing or ambiguous matches fail before writing.
The merger never overwrites the source or an existing output artifact.

## Quantize or estimate

Run a profile-driven streaming conversion:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile path\to\verified-profile.json
```

Estimate storage without reading tensor payloads or writing an output:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    --profile path\to\verified-profile.json `
    --dry-run
```

The estimate prints `estimated_output_bytes`, `estimated_output_mib`, and
`estimated_output_gib`, including the safetensors header and quantization
metadata.

Use `--device cuda` for the supported ConvRot paths when CUDA is available.
Plain `int8` and `int6_rowwise` remain CPU paths; CPU is the default.

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile path\to\verified-profile.json `
    --device cuda
```

Add `--timing` to report conversion stages, action totals, source-size
buckets, and the slowest tensors after conversion:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile path\to\verified-profile.json `
    --device cuda `
    --timing
```

Timing is opt-in and does not change the serialized output. The `write` value
measures `file.write()` time; no flush or `fsync()` is performed.

Merge LoRA files and quantize in one streaming run by repeating paired options:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile path\to\verified-profile.json `
    --adapter-path path\to\style-a.safetensors `
    --adapter-strength 0.65 `
    --adapter-path path\to\style-b.safetensors `
    --adapter-strength 0.40
```

Use a strict TOML configuration when paths and LoRA settings need to be
repeatable:

```powershell
uv run potatoforge quantize --config configs\quantize.toml
```

```toml
[paths]
source = "models/source.safetensors"
profile = "profiles/model/verified.json"
quantized_output = "outputs/quantized.safetensors"

[quantize]
[[quantize.adapters]]
path = "models/style-a.safetensors"
strength = 0.65
```

LoRA paths are relative to the project root unless absolute. Direct CLI
LoRA options override configured LoRA files.

## Weight-only profile workflow

Create a reconstruction and storage report for supported floating-point
two-dimensional `.weight` tensors:

```powershell
uv run potatoforge profile audit `
    path\to\source.safetensors `
    --output reports\weight-audit.json
```

The default candidates are ConvRot W4A4, MSE-optimized ConvRot W4A4, packed
ConvRot INT6, and ConvRot INT8. Use `--device cuda` for supported ConvRot
candidates. Plain INT8 and plain INT6 can be requested explicitly through the
audit implementation, but are not part of the default audit.

Generate a target-size profile directly:

```powershell
uv run potatoforge profile optimize `
    reports\weight-audit.json `
    profiles\generated\model.json `
    --profile-id model-target `
    --target-size-gib 8.5 `
    --method int8_convrot,convrot_w4a4_mse `
    --exclude-prefix blocks.0.,blocks.1.
```

Or use a TOML configuration:

```powershell
uv run potatoforge profile optimize --config configs\optimize.toml
```

`--dry-run` validates and reports the generated choice without writing the
profile. `--max-relative-l2-error`, `--exclude-prefix`, and
`--exclude-suffix` constrain the optimizer. Generated profiles still require
artifact and runtime validation.

Analyze an existing report as a workbook:

```powershell
uv run potatoforge profile analyze `
    --audit reports\weight-audit.json `
    --output reports\weight-analysis.xlsx
```

The workbook precomputes feasible profiles from the smallest allowed output
through all-BF16 at 0.5 GiB intervals by default. Use
`--target-size-step-gib 1.0` to change the interval and
`--target-size-gib 4.5` to select a target. Use `--method` and
`--exclude-prefix` for the recommendation sweep.

Print exact tensor measurements instead of creating a workbook:

```powershell
uv run potatoforge profile analyze `
    --audit reports\weight-audit.json `
    --tensor blocks.12.mlp.down.weight

uv run potatoforge profile analyze `
    --source models\model.safetensors `
    --tensors blocks.11.mlp.down.weight,blocks.12.mlp.down.weight
```

The source mode and the older positional audit input remain supported. Source
mode requires `--tensor` or `--tensors` and does not create a workbook.

## Activation workflow

Activation commands use a calibration pair: JSON metadata plus its adjacent
`.safetensors` statistics file. The cache is reusable and avoids loading the
complete model for later inspection, scoring, comparison, or optimization.

### Build a candidate cache

```powershell
uv run potatoforge activation audit `
    path\to\source.safetensors `
    --activation-calibration calibration\session.json `
    --output reports\activation-audit.json `
    --method convrot_w4a4,convrot_w4a4_mse,int6_convrot,int8_convrot `
    --device cuda `
    --timing
```

The default candidates are `convrot_w4a4`, `convrot_w4a4_mse`,
`int6_convrot`, and `int8_convrot`. Use `--tensor` for a comma-separated
subset of calibration tensors, `--calibration-stats` to select a separate
statistics file, and `--overwrite` to replace the cache pair.

### Inspect and score the cache

Inspect cached activation errors without requantizing weights:

```powershell
uv run potatoforge activation inspect `
    --audit-cache reports\activation-audit.json `
    --activation-calibration calibration\session.json `
    --tensor blocks.27.attn.wo.weight `
    --top-n 5 `
    --output reports\activation-inspection.json
```

Write a standalone score report:

```powershell
uv run potatoforge activation score `
    --audit-cache reports\activation-audit.json `
    --activation-calibration calibration\session.json `
    --metric aggregate_observed_relative_sse `
    --output reports\activation-score.json
```

### Optimize and compare

Generate a profile directly from the cache:

```powershell
uv run potatoforge activation optimize `
    --audit-cache reports\activation-audit.json `
    --activation-calibration calibration\session.json `
    --target-size-gib 8.5 `
    --method convrot_w4a4,convrot_w4a4_mse,int6_convrot,int8_convrot `
    --baseline-method convrot_w4a4 `
    --output profiles\generated\activation.json `
    --summary-output reports\activation-optimize-summary.json
```

The optimizer can instead take `--target-size-mib`,
`--promotion-budget-mib`, or `--promotion-budget-bytes`; exactly one budget
option is required. It validates the calibration pair against the cache and
does not read source payloads or requantize the source checkpoint. Use
`--exclude-prefix` and `--exclude-suffix` to force matching tensors to keep.

Compare metrics and profile choices in one workbook:

```powershell
uv run potatoforge activation compare `
    --audit-cache reports\activation-audit.json `
    --activation-calibration calibration\session.json `
    --target-size-gib 8.5 `
    --output reports\activation-comparison.xlsx `
    --top-n 20
```

Use `--metrics` for a comma-separated metric list, `--method` for the allowed
optimizer methods, and `--baseline-method` to choose the comparison baseline.
The generated profile still requires the normal `quantize` command. Activation
metrics are calibration and reconstruction proxies, not runtime or image-
quality validation.

### Merge calibration pairs

```powershell
uv run potatoforge activation merge `
    calibration\first.json `
    calibration\second.json `
    --output calibration\merged
```

Only compatible calibration pairs can be merged; use `--overwrite` to replace
the output pair.

## Patches and extraction

Write one quantization patch for an exact tensor or a prefix:

```powershell
uv run potatoforge patch `
    path\to\source.safetensors `
    patches\blocks.safetensors `
    --tensor blocks.* `
    --action int8_convrot
```

A name ending in `.*` is a literal prefix selector, so `blocks.*` selects
source names that start with `blocks`. The patch is intended for the
[ComfyUI-PotatoForge custom node](https://github.com/bakapotatolord/ComfyUI-PotatoForge).

Extract a tensor family, optionally renaming its prefix:

```powershell
uv run potatoforge extract `
    path\to\source.safetensors `
    path\to\text-encoder.safetensors `
    --prefix text_model.encoder `
    --output-prefix encoder
```

## Formats and output safety

Profiles can select `keep`, `int8`, `int8_convrot`, `int6_rowwise`,
`int6_convrot`, `convrot_w4a4`, or `convrot_w4a4_mse` per tensor. ConvRot
formats use the runtime-specific group size and storage layout defined by
their format markers. Rules are validated before conversion and applied in
order by prefix and suffix, with an explicit default action.

Conversion, LoRA merging, patching, and extraction refuse source/output
collisions and existing artifacts unless the operation exposes and receives
its overwrite option. The exporter plans the complete output layout from the
header, then processes one tensor at a time.

## Development tests

The public CLI does not include a test command. Contributors can run the
standard-library suite directly:

```powershell
uv run python -m unittest discover -s tests -v
```

## Legacy command names

The former flat names remain accepted as hidden compatibility aliases without
deprecation warnings. New scripts should use the canonical paths above.

| Legacy alias | Canonical command |
| --- | --- |
| `inspect-header` | `inspect model` |
| `inspect-lora` | `inspect lora` |
| `merge-lora` | `lora merge` |
| `audit` | `profile audit` |
| `analyze` | `profile analyze` |
| `optimize` | `profile optimize` |
| `activation-audit` | `activation audit` |
| `activation-inspect` | `activation inspect` |
| `activation-score` | `activation score` |
| `activation-optimize` | `activation optimize` |
| `activation-compare` | `activation compare` |
| `calibration-merge` | `activation merge` |
