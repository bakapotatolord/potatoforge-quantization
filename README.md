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

## Start with the local UI

The local UI is the recommended starting point for inspecting models, building
profiles, and quantizing with configured folders. A built frontend is included;
after installation, launch it with:

```powershell
uv run potatoforge ui
```

`potatoforge ui` starts a local FastAPI service, opens the UI in your browser,
and keeps all model paths on the local machine. After changing frontend source,
rebuild with `pnpm --dir web install` and `pnpm --dir web build`. Use the CLI
below for scripting and advanced workflows.

## CLI and advanced workflows

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

The opt-in `int5_convrot_mse`, `int6_convrot_mse`, `int7_convrot_mse`, and `int8_convrot_mse`
actions keep ordinary ConvRot storage and loader markers while searching lower
row scales. This can lower average reconstruction error but can increase
maximum error by clipping outliers; each nonzero row evaluates 34 candidates.
These actions are not a promise of faster conversion or better generated
images. MSE actions require a runtime compatible with their ordinary format;
INT7 actions require the matching external INT7-capable
`ComfyUI-PotatoForge-INTx` release.

`int8_convrot_mse` preserves ordinary INT8 ConvRot's rotation and quantization
arithmetic, including BF16 rounding for BF16 source weights. It changes only
scale selection; regular `int8_convrot` keeps its existing behavior. Both are
available separately in profiles, weight/activation audits, and profile selection.

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

### Experimental Krea2/Kroma MLP pruning loop

The pruning workflow keeps stage artifacts small and uses original MLP channel
IDs throughout. Build static BF16 weight influence once, select a cumulative
stage from aggregate Comfy calibration data, inspect the companion final-output
cascade audit, and apply only an accepted plan to the original checkpoint:

```powershell
uv run potatoforge prune krea2-mlp weight-stats `
    --model path\to\kroma-bf16.safetensors `
    --output artifacts\krea2-down-weight-stats.safetensors

uv run potatoforge prune krea2-mlp select `
    --activation-stats artifacts\stage1-activation-stats.safetensors `
    --weight-stats artifacts\krea2-down-weight-stats.safetensors `
    --add-channels-per-block 256 `
    --output-plan artifacts\stage1-plan.json

uv run potatoforge prune krea2-mlp audit-summary `
    --audit artifacts\stage1-cascade-audit.json `
    --plan artifacts\stage1-plan.json

uv run potatoforge prune krea2-mlp apply-plan `
    --model path\to\kroma-bf16.safetensors `
    --plan artifacts\stage6-plan.json `
    --output path\to\kroma-pruned-bf16.safetensors
```

`--activation-stats` also accepts a direct (non-recursive) directory of matching
`name.json` and `name.safetensors` calibration pairs. Files are sorted
deterministically, validated for the same model/plan/schema, and combined by
summing raw `sum_squared_activation` and `observation_count` tensors before
computing the mean; per-file means are never averaged. Unrelated files are
ignored.

Repeat calibration and selection with `--base-plan` for later stages. The
local score proposes channels; the final-output cascade audit and fixed visual
tests remain the acceptance gate. Keep prompts, seeds, resolution, sampler,
scheduler, steps, guidance, text encoder, and source quantized model fixed
between stages. Activation stats should come from the unpruned quantized model
used for inference; static weight stats come from the original floating-point
source. This mixed-source score is intentional in V1.

### Experimental Krea2/Kroma whole-block pruning loop

The companion Comfy node writes one calibration JSON/Safetensors pair per run.
Rank a direct, non-recursive directory, audit its shortlist in Comfy, then
explicitly accept one original block ID:

```powershell
uv run potatoforge prune krea2-block rank --calibration "C:\Misc\kroma3-block-pruning\round_0\calibration" --top-k 5 --output "C:\Misc\kroma3-block-pruning\round_0\candidates.json"
uv run potatoforge prune krea2-block audit-summary --audit "C:\Misc\kroma3-block-pruning\round_0\audits" --output "C:\Misc\kroma3-block-pruning\round_0\audit-summary.json"
uv run potatoforge prune krea2-block plan-add --block 17 --block-count 28 --output "C:\Misc\kroma3-block-pruning\round_1\base-plan.json"
uv run potatoforge prune krea2-block plan-add --base-plan "C:\Misc\kroma3-block-pruning\round_1\base-plan.json" --block 23 --output "C:\Misc\kroma3-block-pruning\round_2\base-plan.json"
```

Calibration combines raw sums before calculating
`sum_residual_sq / (sum_input_sq + 1e-12)`; unrelated files are ignored and
artifact paths are deterministically sorted. Recalibrate after every accepted
block. Comfy audit v2 stores derived ratios rather than raw error/baseline
sums, so its cross-file metric is a `forward_count`-weighted
`mean_relative_sse`, not an exact energy-weighted aggregate.

## Command groups

| Goal | Canonical commands | Main artifact |
| --- | --- | --- |
| Inspect headers | `inspect model`, `inspect lora` | Header summaries or JSON reports |
| Merge LoRA files | `lora merge` | Unquantized merged checkpoint |
| Build weight profiles | `profile audit`, `profile optimize`, `profile analyze` | Audit JSON, profile JSON, or analysis workbook |
| Analyze activations | `activation audit`, `activation inspect`, `activation score`, `activation optimize`, `activation compare` | Caches, reports, workbook, or profile |
| Experimental Krea2/Kroma MLP pruning | `prune krea2-mlp weight-stats`, `select`, `audit-summary`, `apply-plan` | Weight stats, cumulative plan, audit summary, or pruned checkpoint |
| Experimental Krea2/Kroma block pruning | `prune krea2-block rank`, `audit-summary`, `plan-add` | Candidate ranking, audit summary, or bypass plan |
| Convert | `quantize` | Profile-driven safetensors checkpoint |
| Patch or extract | `patch`, `extract` | Patch or tensor-family safetensors file |

## More documentation

- [Advanced CLI reference](ADVANCED_CLI.md)
- [ComfyUI-PotatoForge](https://github.com/bakapotatolord/ComfyUI-PotatoForge) for activation calibration and quantization patches
- [ComfyUI-PotatoForge-INTx](https://github.com/bakapotatolord/ComfyUI-PotatoForge-INTx) for INT5/INT6 and ConvRot loading; INT7 artifacts require the corresponding INT7-capable custom-node release

## Kroma GPTQ v1

`convrot_w4a4_mse_gptq` uses model-wide logical-input Gram statistics to
compensate W4A4 rounding errors within each Gram block. Collect and finalize a
BD512 JSON/Safetensors calibration pair with the ComfyUI PotatoForge GPTQ
calibration nodes. Supply the JSON path with `--gptq-calibration`; the
Safetensors file must have the same stem in the same directory. For example:

```text
potatoforge quantize SOURCE.safetensors OUTPUT.safetensors --profile GPTQ_PROFILE.json --gptq-calibration CALIBRATION.json --device cuda --timing
```

Set the profile's action to `convrot_w4a4_mse_gptq` for the desired weights.

The calibration must match the unquantized source model's logical weight names
and shapes. The method shares source Gram preparation across sibling projections,
but optimizes each weight independently. It emits the existing `convrot_w4a4`
runtime payload. Use `--device cuda` to run MSE preparation and the GPTQ solve
on CUDA; prepared factors are reused for sibling weights on the selected device.
The existing W4A4 MSE search selects fixed row scales first; GPTQ then adjusts
the codes using those scales. A valid artifact may declare another block size
that is a multiple of 256. No full Gram is materialized.

With `--config`, add `gptq_calibration = "path/to/calibration.json"` under
`[paths]`; `--gptq-calibration` overrides that value. Calibration is checked
against the source model before checkpoint writing begins. The output keeps
the same packed weights, FP32 scales, `convrot_w4a4` marker, and runtime
metadata as ordinary ConvRot W4A4; load it with the existing W4A4 loader.
The working set includes the current source's Gram and up to two cached factor
sets (each roughly 32 MiB for a 16384-wide source with BD512), rotated
weights, and the existing MSE search and GPTQ solve buffers. The exact peak
depends on the largest output width and PyTorch's temporary allocations.
