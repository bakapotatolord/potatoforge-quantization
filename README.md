# PotatoForge Quants

Python scripts for profile-driven, streaming quantization of
image-generation transformer and DiT safetensors checkpoints.

Conversion is header-first: the complete output layout is planned before
payloads are processed, then tensors are read, quantized or copied, and written
one at a time instead of loading the full checkpoint into memory, primarly for PCs with limited RAM.

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

- ComfyUI-compatible tensorwise INT8 & ConvRot INT8
- Comfy-compatible ConvRot W4A4 with signed packed INT4 weights
- Experimental Rowwise INT6 (`int6_rowwise`) and ConvRot INT6 (`int6_convrot`)
- Header-first streaming conversion, one tensor at a time, to reduce peak RAM usage
- Explicit JSON layer profiles and quantization audits
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
| `inspect-header` | Inspect a safetensors header without reading payloads.                                     |
| `inspect-lora`   | Inspect LoRA adapter structure without reading payloads.                                   |
| `merge-lora`     | Merge one or more adapters into a new checkpoint.                                          |
| `audit`          | Compare INT8, INT6, ConvRot INT8, ConvRot INT6, and W4A4 by size and reconstruction error. |
| `optimize`       | Generate a profile from an audit report and target size; INT6 methods are opt-in.          |
| `quantize`       | Convert a checkpoint with a profile, optionally merging LoRA adapters.                     |
| `extract`        | Extract tensors matching a source prefix.                                                  |
| `test`           | Run the standard-library test suite.                                                       |

The usual profile workflow is `audit` → `optimize` → `quantize`.

### Detailed examples

#### Inspect a safetensors header

```powershell
uv run potatoforge inspect-header path\to\model.safetensors
```

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

#### Choose serial or batched input staging

The default is synchronous whole-tensor RAM staging. Batched mode reads each
complete input batch before quantization and writing:

```powershell
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile profiles\kroma\kroma-v0.1-balanced.json `
    --io-mode batched `
    --input-buffer-gib 4
```

Batched mode is bounded staging only; it does not enable concurrent
prefetching or parallel quantization.

If no input buffer is supplied, batched mode falls back to serial. Use
`--io-mode serial` to force serial processing.

#### Use a quantization TOML config

Repeat a quantization run from a TOML config:

```powershell
uv run potatoforge quantize --config configs\quantize.toml
```

The config stores source, profile, output, I/O settings, and optional adapters.
Use one ordered `[[quantize.adapters]]` table per adapter:

```toml
[quantize]
io_mode = "batched"
input_buffer_gib = 4.0

[[quantize.adapters]]
path = "models/style-a.safetensors"
strength = 0.65
```

Adapter paths are relative to the project root unless absolute. Direct CLI
values remain available as overrides, for example `--io-mode batched` or a
repeat of the adapter options.

#### Generate a target-size profile

Generate a candidate profile from a weight-audit report with the sample config:

```powershell
uv run potatoforge optimize --config configs\optimize.toml
```

TOML configs are currently supported by `optimize` and `quantize`. The other
commands use direct arguments and do not have config files.

#### Audit reconstruction and storage

Compare every BF16 or F16 2-D `.weight` tensor against all supported formats
without a profile. The command prints a storage/error table and writes the
same data as JSON:

```powershell
uv run potatoforge audit `
    path\to\source.safetensors `
    --output reports\weight-audit.json
```

Plain INT6 is unavailable when a layer's input width is not divisible by four;
ConvRot methods are unavailable when it is not divisible by the current group
size of 256. This measures storage and weight reconstruction only, not ComfyUI
runtime compatibility, speed, or image quality.

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
`int6_convrot`, or `convrot_w4a4` independently for each source tensor.
Rules are validated before conversion and applied in order by prefix and
suffix, with an explicit default action. A profile can also convert kept
floating-point tensors to BF16 with `keep_dtype = "BF16"`.

ConvRot formats use the runtime-specific group size and storage layout defined
by their format markers. Packed W4A4 stores two signed INT4 values per byte;
INT6 formats store four values in three bytes.

### Header-first streaming and output safety

The exporter reads only the safetensors header first, plans the complete
output layout, and then reads, transforms, and writes one source tensor at a
time. Batched mode provides bounded whole-tensor staging; serial mode is
available as a fallback and is selected automatically when batched mode has no
input buffer.

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

`audit` compares supported formats for BF16/F16 two-dimensional `.weight`
tensors, reporting per-layer reconstruction error, storage bytes, savings,
and a JSON report. It measures weight reconstruction and storage only; it does
not prove runtime speed, compatibility, or image quality.

`optimize` uses an audit report to generate a target-size profile. It supports
method selection, maximum-error limits, prefix/suffix exclusions, dry runs,
and explicit opt-in for INT6 methods. Generated profiles still require
artifact and runtime validation.

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
