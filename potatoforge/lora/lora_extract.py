"""Extract a LoRA from a fine-tuned checkpoint relative to its base checkpoint.

For every matching tensor the weight delta ``tuned - base`` is computed in the
original (unquantized, unrotated) weight domain and compressed with a
truncated SVD, so that ``base + up @ down ~= tuned``. Nothing here depends on
a particular model: tensors are matched by name (after optional prefix
normalization) and either checkpoint may be a float checkpoint or a ComfyUI
``int8_tensorwise`` / ConvRot checkpoint.

The output uses the ``lora_down`` / ``lora_up`` / ``alpha`` convention that
``lora_merge`` and ComfyUI already read, with ``alpha == rank`` so strength 1.0
applies the full extracted delta.
"""

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypeAlias

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from potatoforge.quantization.hadamard import apply_hadamard_rotation

_FLOAT_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_OUTPUT_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_SIDECAR_SUFFIXES = (".weight_scale", ".comfy_quant")
_QUANT_CONFIG_SUFFIX = ".comfy_quant"
_SUPPORTED_QUANT_FORMAT = "int8_tensorwise"

ExtractProgress: TypeAlias = Callable[[int, int, str], None]


class SplitRule(NamedTuple):
    """Split a fused tensor (last name component ``fused``) into equal row blocks.

    ``SplitRule("qkv", ("q", "k", "v"))`` turns ``x.qkv`` into ``x.q``, ``x.k``
    and ``x.v``, each factored separately. Use it for runtimes that expect
    unfused LoRA keys.
    """

    fused: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionOptions:
    ranks: tuple[int, ...] = (64,)
    # With ``energy`` set, ``ranks`` must hold one value (the per-tensor cap) and
    # each tensor gets the smallest rank capturing that share of the delta energy.
    energy: float | None = None
    key_prefix: str = "diffusion_model."
    base_prefix: str | None = None
    tuned_prefix: str | None = None
    include: str | None = None
    exclude: str | None = None
    split: tuple[SplitRule, ...] = ()
    min_relative_delta: float = 1e-4
    include_diffs: bool = False
    dtype: str = "fp16"
    device: str | None = None
    oversample: int = 32
    power_iterations: int = 6
    seed: int = 0


@dataclass
class TensorReport:
    name: str
    shape: tuple[int, ...]
    relative_delta: float
    delta_norm: float
    residual: dict[str, float] = field(default_factory=dict)
    rank: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    outputs: dict[str, Path]
    tensors: list[TensorReport]
    skipped: dict[str, str]
    unmatched_base: list[str]
    unmatched_tuned: list[str]


def validate_options(options: ExtractionOptions) -> None:
    if not options.ranks or any(rank < 1 for rank in options.ranks):
        raise ValueError("Ranks must be positive integers.")

    if len(set(options.ranks)) != len(options.ranks):
        raise ValueError("Ranks must be unique.")

    if options.energy is not None:
        if not 0.0 < options.energy <= 1.0:
            raise ValueError("Energy target must be in (0, 1].")

        if len(options.ranks) != 1:
            raise ValueError(
                "Energy mode takes exactly one rank value, used as the "
                "per-tensor maximum."
            )

    if options.dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            f"Unsupported output dtype {options.dtype!r}; "
            f"choose from {sorted(_OUTPUT_DTYPES)}."
        )

    if options.min_relative_delta < 0:
        raise ValueError("min_relative_delta must not be negative.")


class _Checkpoint:
    """Lazy reader that yields float32 weights, dequantizing when needed."""

    def __init__(self, path: Path, prefix: str | None) -> None:
        self.path = path
        self._file = safe_open(str(path), framework="pt", device="cpu")
        raw_names = set(self._file.keys())
        self._sidecars = {
            name
            for name in raw_names
            if name.endswith(_SIDECAR_SUFFIXES)
        }
        self.raw_names = raw_names - self._sidecars
        self.prefix = prefix
        self.name_map: dict[str, str] = {}

    def bind_prefix(self, prefix: str) -> None:
        self.prefix = prefix
        self.name_map = {
            _strip_prefix(raw, prefix): raw for raw in self.raw_names
        }

    def is_weight(self, name: str) -> bool:
        """True for float tensors and for quantized weights with a config sidecar."""
        raw = self.name_map[name]

        if (
            raw.endswith(".weight")
            and raw[: -len(".weight")] + _QUANT_CONFIG_SUFFIX in self._sidecars
        ):
            return True

        return self._file.get_slice(raw).get_dtype() in _FLOAT_DTYPES

    def shape_of(self, name: str) -> tuple[int, ...]:
        return tuple(self._file.get_slice(self.name_map[name]).get_shape())

    def read(self, name: str, device: torch.device) -> torch.Tensor:
        raw = self.name_map[name]
        tensor = self._file.get_tensor(raw)
        config_key = (
            raw[: -len(".weight")] + _QUANT_CONFIG_SUFFIX
            if raw.endswith(".weight")
            else None
        )

        if config_key in self._sidecars:
            return self._dequantize(raw, tensor, config_key, device)

        if not tensor.is_floating_point():
            raise ValueError(
                f"{self.path.name}: tensor {raw} has dtype {tensor.dtype} "
                "without a supported quantization config."
            )

        return tensor.to(device=device, dtype=torch.float32)

    def _dequantize(
        self,
        raw: str,
        codes: torch.Tensor,
        config_key: str,
        device: torch.device,
    ) -> torch.Tensor:
        config = json.loads(
            self._file.get_tensor(config_key).numpy().tobytes().decode("utf-8")
        )
        quant_format = config.get("format")

        if quant_format != _SUPPORTED_QUANT_FORMAT:
            raise ValueError(
                f"{self.path.name}: tensor {raw} uses unsupported quantization "
                f"format {quant_format!r}; supported: "
                f"{_SUPPORTED_QUANT_FORMAT!r} (with or without ConvRot)."
            )

        scale_key = raw[: -len(".weight")] + ".weight_scale"

        if scale_key not in self._sidecars:
            raise ValueError(
                f"{self.path.name}: tensor {raw} has no {scale_key}."
            )

        scale = self._file.get_tensor(scale_key).to(device, torch.float32)
        weights = codes.to(device, torch.float32) * scale

        if config.get("convrot"):
            weights = apply_hadamard_rotation(
                weights,
                group_size=int(config["convrot_groupsize"]),
                output_dtype=torch.float32,
            )

        return weights


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def _infer_prefix(
    names: Sequence[str],
    reference: set[str],
) -> str:
    """Prefix that, stripped from ``names``, makes most of them appear in ``reference``."""
    best_prefix = ""
    best_hits = sum(1 for name in names if name in reference)
    candidates: set[str] = set()

    for name in list(names)[:64]:
        if name in reference:
            continue

        for ref in reference:
            if name.endswith("." + ref):
                candidates.add(name[: -len(ref)])
                break

    for prefix in candidates:
        hits = sum(
            1 for name in names if _strip_prefix(name, prefix) in reference
        )

        if hits > best_hits:
            best_prefix, best_hits = prefix, hits

    return best_prefix


def _resolve_prefixes(
    base: _Checkpoint,
    tuned: _Checkpoint,
    options: ExtractionOptions,
) -> None:
    base_names = sorted(base.raw_names)
    tuned_names = sorted(tuned.raw_names)

    base_prefix = options.base_prefix or ""

    if options.tuned_prefix is not None:
        tuned_prefix = options.tuned_prefix
    else:
        reference = {_strip_prefix(name, base_prefix) for name in base_names}
        tuned_prefix = _infer_prefix(tuned_names, reference)

    # The base may be the one carrying the prefix instead.
    if (
        options.base_prefix is None
        and not tuned_prefix
        and options.tuned_prefix is None
    ):
        base_prefix = _infer_prefix(base_names, set(tuned_names))

    base.bind_prefix(base_prefix)
    tuned.bind_prefix(tuned_prefix)


def _top_factors(
    delta: torch.Tensor,
    rank_cap: int,
    options: ExtractionOptions,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U, S, V (columns) for the top ``rank_cap`` singular triplets."""
    rows, cols = delta.shape
    limit = min(rows, cols)
    wanted = min(rank_cap, limit)
    sketch = min(wanted + options.oversample, limit)

    if sketch >= limit:
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        return u, s, vh.T

    return torch.svd_lowrank(delta, q=sketch, niter=options.power_iterations)


def _energy_rank(
    singular_values: torch.Tensor,
    total_energy: float,
    target: float,
    rank_cap: int,
) -> int:
    cumulative = (singular_values.double() ** 2).cumsum(0) / max(total_energy, 1e-30)
    reached = int((cumulative < target).sum().item()) + 1
    return max(1, min(reached, rank_cap, singular_values.numel()))


def _split_targets(
    target: str,
    rows: int,
    rules: Sequence[SplitRule],
) -> list[tuple[str, slice]]:
    prefix, _, last = target.rpartition(".")

    for rule in rules:
        if last != rule.fused:
            continue

        if rows % len(rule.parts) != 0:
            raise ValueError(
                f"{target}: {rows} rows are not divisible into "
                f"{len(rule.parts)} parts."
            )

        size = rows // len(rule.parts)
        return [
            (f"{prefix}.{part}" if prefix else part, slice(i * size, (i + 1) * size))
            for i, part in enumerate(rule.parts)
        ]

    return [(target, slice(0, rows))]


def parse_split_rule(text: str) -> SplitRule:
    """Parse ``FUSED:PART,PART[,PART]`` (for example ``qkv:q,k,v``)."""
    fused, separator, parts = text.partition(":")
    names = tuple(part.strip() for part in parts.split(",") if part.strip())

    if not separator or not fused.strip() or len(names) < 2:
        raise ValueError(
            f"Split rule must look like FUSED:PART,PART[,PART]; got {text!r}."
        )

    return SplitRule(fused.strip(), names)


def output_label(rank: int, options: ExtractionOptions) -> str:
    return f"e{options.energy * 100:g}" if options.energy is not None else str(rank)


def resolve_output_paths(
    output_template: str | Path,
    options: ExtractionOptions,
) -> dict[str, Path]:
    """Map each output label to its path, substituting ``{rank}`` in the template."""
    template = str(output_template)

    if len(options.ranks) > 1 and "{rank}" not in template:
        raise ValueError(
            "Multiple ranks require '{rank}' in the output path."
        )

    return {
        output_label(rank, options): Path(
            template.replace("{rank}", output_label(rank, options))
        )
        for rank in options.ranks
    }


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    output_template: str | Path,
    options: ExtractionOptions = ExtractionOptions(),
    on_tensor_started: ExtractProgress | None = None,
) -> ExtractionResult:
    """Write one LoRA per requested rank; ``{rank}`` in the template is substituted."""
    validate_options(options)
    base_file = Path(base_path)
    tuned_file = Path(tuned_path)

    for path in (base_file, tuned_file):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    if base_file.resolve() == tuned_file.resolve():
        raise ValueError("Base and tuned checkpoints must be different files.")

    labels = {rank: output_label(rank, options) for rank in options.ranks}
    outputs = resolve_output_paths(output_template, options)

    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

        partial = output.with_name(output.name + ".partial")

        if partial.exists():
            raise FileExistsError(f"Partial output already exists: {partial}")

        if output.resolve() in (base_file.resolve(), tuned_file.resolve()):
            raise ValueError("Output must differ from the input checkpoints.")

    include = re.compile(options.include) if options.include else None
    exclude = re.compile(options.exclude) if options.exclude else None
    device = torch.device(
        options.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dtype = _OUTPUT_DTYPES[options.dtype]
    rank_cap = max(options.ranks)

    base = _Checkpoint(base_file, options.base_prefix)
    tuned = _Checkpoint(tuned_file, options.tuned_prefix)
    _resolve_prefixes(base, tuned, options)

    shared = [name for name in base.name_map if name in tuned.name_map]

    if not shared:
        raise ValueError(
            "The checkpoints share no tensor names. Set --base-prefix / "
            "--tuned-prefix if their key prefixes differ."
        )

    torch.manual_seed(options.seed)
    tensors: dict[str, dict[str, torch.Tensor]] = {label: {} for label in labels.values()}
    reports: list[TensorReport] = []
    skipped: dict[str, str] = {}

    def emit_pair(
        target: str,
        delta: torch.Tensor,
        relative: float,
        report_name: str,
        shape: tuple[int, ...],
    ) -> None:
        u, s, v = _top_factors(delta, rank_cap, options)
        total_energy = float((delta.double() ** 2).sum().item())
        report = TensorReport(
            report_name,
            shape,
            relative,
            math.sqrt(total_energy),
        )

        for requested in options.ranks:
            label = labels[requested]

            if options.energy is not None:
                rank = _energy_rank(s, total_energy, options.energy, rank_cap)
            else:
                rank = min(requested, s.numel())

            root = s[:rank].sqrt()
            up = u[:, :rank] * root
            down = (v[:, :rank] * root).T
            base_key = f"{options.key_prefix}{target}"
            tensors[label][f"{base_key}.lora_up.weight"] = up.to(out_dtype).contiguous().cpu()
            tensors[label][f"{base_key}.lora_down.weight"] = down.to(out_dtype).contiguous().cpu()
            tensors[label][f"{base_key}.alpha"] = torch.tensor(float(rank))
            residual = (delta - up @ down).norm() / delta.norm().clamp(min=1e-30)
            report.residual[label] = float(residual.item())
            report.rank[label] = rank

        reports.append(report)

    for index, name in enumerate(shared, start=1):
        if on_tensor_started is not None:
            on_tensor_started(index, len(shared), name)

        if include is not None and not include.search(name):
            continue

        if exclude is not None and exclude.search(name):
            continue

        if not (base.is_weight(name) and tuned.is_weight(name)):
            skipped[name] = "not a float or quantized weight tensor"
            continue

        base_shape = base.shape_of(name)
        tuned_shape = tuned.shape_of(name)

        if base_shape != tuned_shape:
            skipped[name] = f"shape mismatch {base_shape} vs {tuned_shape}"
            continue

        base_weight = base.read(name, device)
        delta = tuned.read(name, device) - base_weight
        relative = float(
            (delta.norm() / base_weight.norm().clamp(min=1e-30)).item()
        )

        if not math.isfinite(relative):
            skipped[name] = "non-finite delta"
            continue

        if relative <= options.min_relative_delta:
            skipped[name] = f"unchanged (relative delta {relative:.2e})"
            continue

        if len(base_shape) == 2 and name.endswith(".weight"):
            target = name[: -len(".weight")]

            for part_target, rows in _split_targets(target, base_shape[0], options.split):
                emit_pair(
                    part_target,
                    delta[rows].contiguous(),
                    relative,
                    part_target,
                    base_shape,
                )
        elif options.include_diffs:
            diff_key = f"{options.key_prefix}{name}.diff"

            for label in labels.values():
                tensors[label][diff_key] = delta.to(out_dtype).contiguous().cpu()

            reports.append(
                TensorReport(name, base_shape, relative, float(delta.norm().item()))
            )
        else:
            skipped[name] = "changed but not a 2D weight (use include_diffs)"

    if not reports:
        raise ValueError("No tensors differ between the checkpoints; nothing to extract.")

    for rank in options.ranks:
        label = labels[rank]
        metadata = {
            "extractor": "potatoforge lora extract",
            "rank": "adaptive" if options.energy is not None else str(rank),
            "base": base_file.name,
            "tuned": tuned_file.name,
        }

        if options.energy is not None:
            metadata["energy"] = str(options.energy)

        output = outputs[label]
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        save_file(tensors[label], str(partial), metadata=metadata)
        partial.rename(output)

    return ExtractionResult(
        outputs=outputs,
        tensors=reports,
        skipped=skipped,
        unmatched_base=sorted(n for n in base.name_map if n not in tuned.name_map),
        unmatched_tuned=sorted(n for n in tuned.name_map if n not in base.name_map),
    )


def summarize(result: ExtractionResult) -> dict[str, object]:
    """Aggregate energy captured per output (1 - weighted residual energy)."""
    summary: dict[str, object] = {}

    for label, path in result.outputs.items():
        weighted_total = 0.0
        weighted_residual = 0.0

        for report in result.tensors:
            if label not in report.residual:
                continue

            energy = report.delta_norm ** 2
            weighted_total += energy
            weighted_residual += energy * report.residual[label] ** 2

        summary[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "tensors": sum(1 for r in result.tensors if label in r.residual),
            "energy_captured": 1.0 - weighted_residual / weighted_total if weighted_total else 1.0,
        }

    return summary
