# PotatoForge Quants

Python scripts for profile-driven, streaming quantization of
image-generation transformer and DiT safetensors checkpoints.

Conversion is header-first: the complete output layout is planned before
payloads are processed, then tensors are read, quantized or copied, and written
one at a time instead of loading the full checkpoint into memory, primarily for PCs with limited RAM.

## Quick start

From the repository root, install the project and run a profile-driven
conversion:

```powershell
uv sync
uv run potatoforge quantize `
    path\to\source.safetensors `
    quantized.safetensors `
    --profile profiles\kroma\kroma-v0.1-balanced.json
```

Replace `path\to\source.safetensors` with your checkpoint path. See [Setup](#setup)
for a Python and `pip` installation, or [Usage](#usage) for audits, profile
optimization, LoRA merging, and other commands.

## What it can do

- ComfyUI-compatible tensorwise INT8 and ConvRot INT8
- Comfy-compatible ConvRot W4A4 with signed packed INT4 weights and
  MSE-optimized weight scales
- Experimental Rowwise INT6 (`int6_rowwise`) and ConvRot INT6 (`int6_convrot`)
- Header-first streaming conversion, one tensor at a time, to reduce peak RAM usage
- Explicit JSON layer profiles and quantization audits
- Activation-aware calibration scoring and mixed-precision profile generation
- Header-first LoRA merging with additive tensor-delta support
- Target-size profile optimization
- Strict TOML configuration for repeatable quantization and optimization
- Prefix-based tensor extraction

## Disclosure

The project is highly experimental, I utilized this project to learn more about diffusion models, quantizations and how safetensors work in an iterative manner. AI has been utilized heavily for guidance, benchmarks and writing code. I actually ended up learning a great deal from this so pretty proud with being productive for once. Anyway, back to you AI!

Requires Python 3.11+. Choose either `uv` or Python with `pip`.

## Setup

With `uv`:

```powershell
uv sync
uv run potatoforge --help
```

With Python and `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
potatoforge --help
```

The Python setup uses the standard project metadata in `pyproject.toml`; no
`requirements.txt` file is needed. For the same CUDA 13.2 Torch build used by
the `uv` configuration, install Torch from its index before the project:

```powershell
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e .
```

For another CPU or CUDA target, use that target's official Torch install
command instead. After the Python setup, replace `uv run potatoforge` with
`potatoforge` and `uv run python` with `python` in the commands below.

## Usage

Start with `uv run potatoforge --help` (or `potatoforge --help` in a Python
environment) to see all available options.

### Commands at a glance

| Command          | Purpose                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `inspect-header` | Inspect a safetensors header without reading payloads; optionally show quantization details. |
| `inspect-lora`   | Inspect LoRA adapter structure without reading payloads.                                   |
| `merge-lora`     | Merge one or more adapters into a new checkpoint.                                          |
| `audit`          | Compare INT8, INT6, ConvRot INT8, ConvRot INT6, W4A4, and W4A4 MSE by size and reconstruction error. |
| `activation-score` | Score a calibration against a reusable activation probe cache.                              |
| `calibration-merge` | Merge compatible activation calibration pairs.                                           |
| `profile-from-activation` | Generate a W4A4-baseline mixed profile from activation scores.                    |
| `analyze`        | Render an existing audit as an Excel workbook or inspect one exact tensor.                  |
| `optimize`       | Generate a profile from an audit report and target size; list INT6 methods to use them.    |
| `quantize`       | Convert a checkpoint with a profile, optionally merging LoRA adapters.                     |
| `patch`                    | Generate one quantization patch for an exact tensor or prefix.                                |
| `extract`        | Extract tensors matching a source prefix.                                                  |
| `test`           | Run the standard-library test suite.                                                       |

The usual profile workflow is `audit` → `optimize` → `quantize`.

### Detailed examples

#### Inspect a safetensors header

```powershell
uv run potatoforge inspect-header path\to\model.safetensors
```

Show the quantization summary and each quantized layer:

```powershell
uv run potatoforge inspect-header path\to\quantized.safetensors --quantization
```

Normal quantized outputs record the summary in `__metadata__` and store a JSON
layer map in `potatoforge.quantization_layers`.

#### Inspect a LoRA adapter

Inspect a LoRA adapter without loading tensor payloads:

```powershell
uv run potatoforge inspect-lora `
    path\to\adapter.safetensors `
    --output reports\adapter-inventory.json
```

#### Merge one or more LoRA adapters

Merge one or more adapters into a new checkpoint while preserving source
dtypes:

```powershell
uv run potatoforge merge-lora `
    path\to\source.safetensors `
    path\to\merged.safetensors `
    --adapter-path path\to\style-a.safetensors `
    --adapter-strength 0.65 `
    --adapter-path path\to\style-b.safetensors `
    --adapter-strength 0.40
```

Adapter targets are resolved automatically: exact source names win first;
otherwise the merger accepts one unique dot-boundary suffix match. It fails
before writing when no source tensor or more than one source tensor matches.

The merger never overwrites the source or an existing output artifact.

#### Run a profile-driven streaming conversion

Run a profile-driven streaming conversion:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile profiles\kroma\kroma-v0.1-balanced.json
```

Estimate the final safetensors storage without reading tensor payloads or
writing an output file:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    --profile profiles\kroma\kroma-v0.1-balanced.json `
    --dry-run
```

The command prints `estimated_output_bytes`, `estimated_output_mib`, and
`estimated_output_gib` after applying the profile, including the safetensors
header and quantization metadata.

#### Generate a quantization patch

Generate one patch file from an exact source tensor or a prefix. A name ending
in `.*` uses prefix matching, so `blocks.*` selects every source tensor whose
name starts with `blocks`.

```powershell
uv run potatoforge patch `
    path\to\source.safetensors `
    patches\blocks.safetensors `
    --tensor blocks.* `
    --action int8_convrot
```

The command writes one `.safetensors` patch containing the selected quantized
weight families. It is designed for use with the companion
[ComfyUI-PotatoForge custom node](https://github.com/bakapotatolord/ComfyUI-PotatoForge).

#### Merge LoRA adapters and quantize in one run

Merge LoRA adapters and quantize in the same run:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile profiles\kroma\kroma-v0.1-balanced.json `
    --adapter-path path\to\style-a.safetensors `
    --adapter-strength 0.65 `
    --adapter-path path\to\style-b.safetensors `
    --adapter-strength 0.40
```

Each adapter path is paired with the strength at the same option position.
Adapters are loaded into CPU memory once; the large source checkpoint remains
streamed through the profile quantization pass.

#### Use a quantization TOML config

Repeat a quantization run from a TOML config:

```powershell
uv run potatoforge quantize --config configs\quantize.toml
```

The config stores source, profile, output, and optional adapters.
Use one ordered `[[quantize.adapters]]` table per adapter:

```toml
[quantize]
[[quantize.adapters]]
path = "models/style-a.safetensors"
strength = 0.65
```

Adapter paths are relative to the project root unless absolute. Direct CLI
adapter options remain available as overrides.

#### Generate a target-size profile

Generate a candidate profile from a weight-audit report with the sample config:

```powershell
uv run potatoforge optimize --config configs\optimize.toml
```

TOML configs are currently supported by `optimize` and `quantize`. The other
commands use direct arguments and do not have config files.

#### Audit reconstruction and storage

Compare every BF16, F16, or F32 2-D `.weight` tensor against the four default
ConvRot formats, without a profile. The command prints a storage/error table
and writes the same data as JSON:

```powershell
uv run potatoforge audit `
    path\to\source.safetensors `
    --output reports\weight-audit.json
```

Plain INT8 and INT6 are not part of the default audit. ConvRot methods are
unavailable when a layer's input width is not divisible by the current group
size of 256. This measures storage and weight reconstruction only, not ComfyUI
runtime compatibility, speed, or image quality.

#### Activation-aware profiling

Use a compatible activation calibration pair (JSON metadata plus its adjacent
`.safetensors` statistics file) to rank W4A4 layers by measured output impact.
First create a reusable INT8 ConvRot-to-W4A4 probe cache from the source model:

```powershell
uv run potatoforge audit `
    models\source.safetensors `
    --output reports\w4a4-audit.json `
    --method convrot_w4a4 `
    --activation-calibration calibration\session.json `
    --activation-probe-output reports\w4a4-probe.json
```

The probe cache uses the source checkpoint to generate the reference and
requires calibration whose baseline is INT8 ConvRot.

Score a calibration session against that cache:

```powershell
uv run potatoforge activation-score `
    --probe-cache reports\w4a4-probe.json `
    --activation-calibration calibration\session.json `
    --output reports\activation-score.json
```

Use `calibration-merge` to combine compatible sessions before scoring when
needed. Then generate a final-size profile and run the normal `quantize`
command with it:

```powershell
uv run potatoforge profile-from-activation `
    --activation-score reports\activation-score.json `
    --weight-audit reports\w4a4-audit.json `
    --target-size-gib 8.5 `
    --output profiles\generated\activation-relative.json
```

The generated profile uses ConvRot W4A4 as its baseline and promotes eligible
`^blocks\.` tensors to INT8 ConvRot within the size budget. Use
`--include-regex` to change eligibility or `--promotion-budget-mib` to budget
storage above the W4A4 baseline. Pass `--activation-calibration` to `audit`,
or to source-mode `analyze`, when the audit or tensor report should include the
activation measurements. The default promotion metric is `relative_output_sse`,
the squared normalized output perturbation; use `--metric relative_output_error`
to reproduce the older unsquared objective. These are calibration and
reconstruction proxies; runtime and image-quality validation remain separate.

#### Analyze an audit

Render the existing audit measurements without rerunning quantization:

```powershell
uv run potatoforge analyze `
    --audit reports\weight-audit.json `
    --output reports\weight-analysis.xlsx
```

The workbook now precomputes feasible profiles from the smallest allowed
output through all-BF16, using 0.5 GiB intervals by default. Change the
interval with `--target-size-step-gib 1.0`; `--target-size-gib 4.5` selects
that target in the workbook (and adds it when it is between interval points).
The `Recommendations` sheet has a target-profile dropdown, the `Profiles`
sheet has the static helper table, and `Trade-offs` charts estimated output
size against P95 tensor relative L2. Profiles report raw and normalized
Reconstruction SSE, while `Profile Data` also shows each tensor's weight-energy
fraction; the selected recommendation table shows a read-only next-method
upgrade candidate and its SSE-reduction-per-MiB score.
P95 means 95% of audited tensors have error at or below that value; the maximum
error column exposes the worst tensor separately.
Use `--method int8_convrot,convrot_w4a4_mse` and `--exclude-prefix
blocks.0.,blocks.1.` to keep matching layers in BF16. The chart and errors are
audit proxies only; they do not validate runtime behavior or image quality. Or use
`--tensor blocks.12.mlp.down.weight` to print one exact tensor, or
`--tensors blocks.11.mlp.down.weight,blocks.12.mlp.down.weight` to print
several, without creating a workbook.

To inspect one tensor directly from the source model without auditing every
weight, use:

```powershell
uv run potatoforge analyze `
    --source models\model.safetensors `
    --tensor blocks.12.mlp.down.weight
```

Source mode prints the selected tensors' measured method errors and does not
create a workbook or target-size recommendation. `--audit` and `--source` are
mutually exclusive; the older positional audit path remains accepted for
compatibility.

#### Extract a tensor family

Extract tensors matching a source prefix, optionally renaming the prefix in
the output file:

```powershell
uv run potatoforge extract `
    path\to\source.safetensors `
    path\to\text-encoder.safetensors `
    --prefix text_model.encoder `
    --output-prefix encoder
```

#### Run the test suite

Run the complete test suite:

```powershell
uv run python -m unittest discover -s tests -v
```

## Core feature details

### Quantization formats and profiles

Profiles can select `keep`, `int8`, `int8_convrot`, `int6_rowwise`,
`int6_convrot`, `convrot_w4a4`, or `convrot_w4a4_mse` independently for each
source tensor. W4A4 MSE writes the standard W4A4 runtime family and chooses
scales by reconstruction error.
Rules are validated before conversion and applied in order by prefix and
suffix, with an explicit default action. A profile can also convert kept
floating-point tensors to BF16 with `keep_dtype = "BF16"`.
A rule may optionally specify one `fallback` quantization method. It is tried
only after the rule matches and its primary action fails eligibility for that
tensor; if the fallback is also ineligible, the existing profile-wide default
behavior is used. For example:

```json
{
  "default": "keep",
  "rules": [
    {
      "action": "int8_convrot",
      "fallback": "int8",
      "prefix": "model.diffusion_model.",
      "suffixes": [".weight"]
    }
  ]
}
```

This uses `int8_convrot` where eligible, then `int8` where the primary method
is ineligible, and otherwise uses the profile-wide behavior (`keep`). Fallback
is one level only and does not catch runtime quantization exceptions.

ConvRot formats use the runtime-specific group size and storage layout defined
by their format markers. Packed W4A4 stores two signed INT4 values per byte;
INT6 formats store four values in three bytes.

### Header-first streaming and output safety

The exporter reads only the safetensors header first, plans the complete
output layout, and then reads, transforms, and writes one source tensor at a
time.

The writer validates tensor names, order, offsets, and byte counts as payloads
arrive. Conversion, LoRA merging, and extraction refuse source/output
collisions and existing artifacts. JSON report directories are created when
needed.

### LoRA inspection, merging, and fused quantization

`inspect-lora` reports adapter pairs, ranks, additive `.diff` tensors, and
unsupported contracts without reading payloads. The merger supports standard
two-factor linear LoRA and additive tensor deltas, multiple adapters with
independent strengths, and several common adapter naming conventions.

Targets use an exact source-name match first, then one unique dot-boundary
suffix match. Ambiguous or missing matches fail before output is written.
Adapters are loaded into CPU memory once, while `quantize` can apply them
inside the streaming conversion so no intermediate merged checkpoint is
needed.

### Auditing and profile optimization

`audit` compares supported formats for BF16/F16/F32 two-dimensional `.weight`
tensors, reporting per-layer reconstruction error, storage bytes, savings,
and a JSON report. It measures weight reconstruction and storage only; it does
not prove runtime speed, compatibility, or image quality.

`optimize` uses an audit report to generate a target-size profile. It supports
method selection, maximum-error limits, prefix/suffix exclusions, dry runs,
and INT6 methods when they are listed in `methods`. Generated profiles still
require artifact and runtime validation.

CLI method selections and prefix/suffix exclusions are comma-separated; TOML
configurations use arrays.

### TOML configuration and tensor extraction

`quantize` and `optimize` accept strict version-one TOML configurations with
project-root-relative paths. Quantization configs support I/O settings and
ordered repeated `[[quantize.adapters]]` tables; direct CLI values can override
config values.

`extract` copies tensors matching a source prefix into a new safetensors file,
optionally renaming the extracted prefix while preserving source metadata.

## Layout

```text
potatoforge/   reusable conversion library
profiles/      model-family JSON policies
tests/         standard-library test suite
```

## ComfyUI Extensions

- [ComfyUI-PotatoForge-INT6](https://github.com/bakapotatolord/ComfyUI-PotatoForge-INT6) enables ComfyUI to load INT6 and INT6 ConvRot quantized models.

## Credits

The following repositories were used as references for this project:

- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [bedovyy/comfy-dit-quantizer](https://github.com/bedovyy/comfy-dit-quantizer)
