# PotatoForge Quants

Profile-driven, header-first streaming quantization tools for image-generation
transformer and DiT safetensors checkpoints. Conversion reads and processes one
tensor at a time, reducing peak RAM use on ordinary PCs.

## Experimental status

PotatoForge is experimental and was built iteratively for learning about
diffusion models, quantization, and safetensors. Audits, estimates, and
successful conversions report tensor-level artifacts and reconstruction or
storage proxies; they do not prove runtime compatibility, speed, or image
quality. Use a profile only when you have independently verified that it
matches the source, then test the output in the intended ComfyUI runtime and
loader.

## Setup

Requires Python 3.11+. With `uv`:

```powershell
uv sync
uv run potatoforge --help
```

With an existing Python installation and `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
potatoforge --help
```

Use the Torch build appropriate for the target CPU or CUDA environment. After
the `pip` setup, replace `uv run potatoforge` with `potatoforge` below.

## Choose your path

### Use a profile you have verified

Inspect the input, optionally estimate storage, quantize, and inspect the
output:

```powershell
# Show the source tensors and metadata before changing anything.
uv run potatoforge inspect model path\to\source.safetensors

# Estimate the output size without writing a checkpoint.
uv run potatoforge quantize `
    path\to\source.safetensors `
    --profile path\to\verified-profile.json `
    --dry-run

# Write the quantized checkpoint with the verified profile.
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile path\to\verified-profile.json

# Confirm the quantization metadata in the output checkpoint.
uv run potatoforge inspect model path\to\quantized.safetensors --quantization
```

Do not treat a profile as a generic match: use it only when its source and
loader compatibility applies to the checkpoint.

### Create a weight-only profile

Create an audit, optimize a target-size profile, then run the normal
conversion:

```powershell
# Measure storage and reconstruction error for supported weight formats.
uv run potatoforge profile audit `
    path\to\source.safetensors `
    --output reports\weight-audit.json

# Generate a profile that targets the requested output size.
uv run potatoforge profile optimize `
    reports\weight-audit.json `
    profiles\generated\model.json `
    --profile-id model-target `
    --target-size-gib 8.5

# Write a checkpoint using the generated profile.
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile profiles\generated\model.json
```

### Create an activation-aware profile

Build an activation candidate cache, optimize from that cache, then quantize.
This is the [advanced activation path](ADVANCED_CLI.md#activation-workflow):

```powershell
# Build a reusable cache from the source model and calibration pair.
uv run potatoforge activation audit `
    path\to\source.safetensors `
    --activation-calibration calibration\session.json `
    --output reports\activation-audit.json

# Generate a target-size profile from the activation cache.
uv run potatoforge activation optimize `
    --audit-cache reports\activation-audit.json `
    --activation-calibration calibration\session.json `
    --target-size-gib 8.5 `
    --output profiles\generated\activation.json

# Write a checkpoint using the activation-aware profile.
uv run potatoforge quantize `
    path\to\source.safetensors `
    path\to\quantized.safetensors `
    --profile profiles\generated\activation.json
```

### Create a ComfyUI patch

Create one exact-tensor or prefix patch and apply it with the companion
[ComfyUI-PotatoForge custom node](https://github.com/bakapotatolord/ComfyUI-PotatoForge):

```powershell
# Write one patch containing matching tensors for the companion custom node.
uv run potatoforge patch `
    path\to\source.safetensors `
    patches\blocks.safetensors `
    --tensor blocks.* `
    --action int8_convrot
```

`blocks.*` is a literal prefix selector: it selects source tensor names that
start with `blocks`.

## Command groups

| Goal | Canonical commands | Main artifact |
| --- | --- | --- |
| Inspect headers | `inspect model`, `inspect lora` | Header summaries or JSON reports |
| Merge LoRA files | `lora merge` | Unquantized merged checkpoint |
| Build weight profiles | `profile audit`, `profile optimize`, `profile analyze` | Audit JSON, profile JSON, or analysis workbook |
| Analyze activations | `activation audit`, `activation inspect`, `activation score`, `activation optimize`, `activation compare`, `activation merge` | Caches, reports, workbook, or profile |
| Convert | `quantize` | Profile-driven safetensors checkpoint |
| Patch or extract | `patch`, `extract` | Patch or tensor-family safetensors file |

## More documentation

- [Advanced CLI reference](ADVANCED_CLI.md)
- [ComfyUI-PotatoForge](https://github.com/bakapotatolord/ComfyUI-PotatoForge) for activation calibration and quantization patches
- [ComfyUI-PotatoForge-INT6](https://github.com/bakapotatolord/ComfyUI-PotatoForge-INT6) for INT6 and INT6 ConvRot loading
