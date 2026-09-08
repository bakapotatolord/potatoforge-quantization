from collections.abc import Iterable

import torch


def calculate_linear_lora_delta(
    down: torch.Tensor,
    up: torch.Tensor,
    strength: float,
    alpha: float | None = None,
) -> torch.Tensor:
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("LoRA factors must be rank-2 tensors.")

    rank, input_features = down.shape
    output_features, up_rank = up.shape

    if rank <= 0 or up_rank != rank:
        raise ValueError(
            "LoRA factor ranks must match and be positive."
        )

    if input_features <= 0 or output_features <= 0:
        raise ValueError(
            "LoRA feature dimensions must be positive."
        )

    scale = strength

    if alpha is not None:
        scale *= alpha / rank

    return scale * (
        up.to(dtype=torch.float32)
        @ down.to(dtype=torch.float32)
    )

def calculate_additive_tensor_delta(
    delta: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if not delta.is_floating_point():
        raise ValueError("Additive delta must be floating-point.")

    return strength * delta.to(dtype=torch.float32)


def reconstruct_lokr_direct(
    w1: torch.Tensor,
    w2: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError(
            "Direct LoKr currently supports only 2D lokr_w1/lokr_w2 "
            f"(got w1={tuple(w1.shape)}, w2={tuple(w2.shape)})"
        )

    delta = torch.kron(
        w1.to(dtype=torch.float32),
        w2.to(dtype=torch.float32),
    )

    if tuple(delta.shape) != tuple(target_shape):
        raise ValueError(
            "LoKr shape mismatch: "
            f"w1={tuple(w1.shape)}, "
            f"w2={tuple(w2.shape)}, "
            f"produced={tuple(delta.shape)}, "
            f"target={tuple(target_shape)}"
        )

    return delta


def merge_tensor_contributions(
    base: torch.Tensor,
    contributions: Iterable[torch.Tensor],
) -> torch.Tensor:
    if not base.is_floating_point():
        raise ValueError("Base tensor must be floating-point.")

    merged = base.to(dtype=torch.float32)

    for contribution in contributions:
        if contribution.shape != base.shape:
            raise ValueError(
                "Contribution shape must match base tensor."
            )

        if contribution.device != base.device:
            raise ValueError(
                "Contribution device must match base tensor."
            )

        if not contribution.is_floating_point():
            raise ValueError(
                "Tensor contributions must be floating-point."
            )

        merged = merged + contribution.to(dtype=torch.float32)
        del contribution

    return merged.to(dtype=base.dtype)
