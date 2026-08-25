from typing import Literal, TypedDict

from potatoforge.headers.source_header import SourceModelHeader

class PairConvention(TypedDict):
    name: str
    down_suffix: str
    up_suffix: str

class PairCandidate(TypedDict):
    convention_name: str
    target: str
    down_key: str
    up_key: str

class LinearPairResult(TypedDict):
    rank: int
    output_features: int
    input_features: int

class DiscoveredLinearPair(TypedDict):
    convention_name: str
    target: str
    down_key: str
    up_key: str
    rank: int
    input_features: int
    output_features: int

class PairDiscoveryResult(TypedDict):
    pairs: list[DiscoveredLinearPair]
    unpaired_down_keys: list[str]
    unpaired_up_keys: list[str]

class DiscoveredAdditiveDelta(TypedDict):
    key: str
    target: str
    shape: list[int]
    dtype: str

AdapterTensorKind = Literal[
    "additive_tensor_delta",
    "alpha",
    "unsupported",
]

AdapterContract = Literal[
    "linear_lora",
    "additive_tensor_delta",
    "alpha",
    "loha",
    "lokr",
    "oft",
    "boft",
    "oft_or_boft",
    "dora",
    "weight_norm",
    "bias_norm",
    "additive_bias_delta",
    "conv_lora",
    "reshape",
    "set_weight",
    "unsupported",
]

class AdapterTensorClassification(TypedDict):
    kind: AdapterTensorKind
    target: str | None
    contract: AdapterContract

AdapterInventoryKind = Literal[
    "linear_down",
    "linear_up",
    "additive_tensor_delta",
    "alpha",
    "unpaired_down",
    "unpaired_up",
    "unsupported",
]

class AdapterTensorRecord(TypedDict):
    key: str
    shape: list[int]
    dtype: str
    kind: AdapterInventoryKind
    target: str | None
    contract: AdapterContract


class AdapterInspectionResult(TypedDict):
    pairs: list[DiscoveredLinearPair]
    additive_deltas: list[DiscoveredAdditiveDelta]
    tensors: list[AdapterTensorRecord]

PAIR_CONVENTIONS: tuple[PairConvention, ...] = (
    {
        "name": "comfy_up_down",
        "down_suffix": ".lora_down.weight",
        "up_suffix": ".lora_up.weight",
    },
    {
        "name": "diffusers_ab",
        "down_suffix": ".lora_A.weight",
        "up_suffix": ".lora_B.weight",
    },
    {
        "name": "bare_ab",
        "down_suffix": ".lora_A",
        "up_suffix": ".lora_B",
    },
    {
        "name": "diffusers_legacy_ab",
        "down_suffix": "_lora.down.weight",
        "up_suffix": "_lora.up.weight",
    },
    {
        "name": "diffusers_dotted_ab",
        "down_suffix": ".lora.down.weight",
        "up_suffix": ".lora.up.weight",
    },
    {
        "name": "transformers_lora_linear_layer",
        "down_suffix": ".lora_linear_layer.down.weight",
        "up_suffix": ".lora_linear_layer.up.weight",
    },
    {
        "name": "qwen_default_ab",
        "down_suffix": ".lora_A.default.weight",
        "up_suffix": ".lora_B.default.weight",
    },
)


_UNSUPPORTED_CONTRACT_SUFFIXES: tuple[
    tuple[str, AdapterContract], ...
] = (
    (".hada_w1_a", "loha"),
    (".hada_w1_b", "loha"),
    (".hada_w2_a", "loha"),
    (".hada_w2_b", "loha"),
    (".hada_t1", "loha"),
    (".hada_t2", "loha"),
    (".lokr_w1", "lokr"),
    (".lokr_w2", "lokr"),
    (".lokr_w1_a", "lokr"),
    (".lokr_w1_b", "lokr"),
    (".lokr_w2_a", "lokr"),
    (".lokr_w2_b", "lokr"),
    (".lokr_t2", "lokr"),
    (".oft_blocks", "oft_or_boft"),
    (".rescale", "oft_or_boft"),
    (".dora_scale", "dora"),
    (".lora_magnitude_vector", "dora"),
    (".w_norm", "weight_norm"),
    (".b_norm", "bias_norm"),
    (".diff_b", "additive_bias_delta"),
    (".lora_mid.weight", "conv_lora"),
    (".reshape_weight", "reshape"),
    (".set_weight", "set_weight"),
)

def _read_shape(header: SourceModelHeader, key: str) -> list[int]:
    descriptor = header.tensors.get(key)

    if descriptor is None:
        raise ValueError(f"{key} is missing from headers")

    shape = descriptor["shape"]

    return shape

def inspect_linear_pair(
    header: SourceModelHeader,
    down_key: str,
    up_key: str
) -> LinearPairResult:
    down_shape = _read_shape(header, down_key)
    up_shape = _read_shape(header, up_key)

    if len(down_shape) != 2 or len(up_shape) != 2:
        raise ValueError("LoRA factors must be rank-2 tensors.")

    down_rank, input_features = down_shape
    output_features, up_rank = up_shape

    if down_rank <= 0 or up_rank <= 0:
        raise ValueError("LoRA rank must be a positive integer.")

    if down_rank != up_rank:
        raise ValueError("Ranks do not match")

    if input_features <= 0 or output_features <= 0:
        raise ValueError("LoRA feature dimensions must be positive.")

    return LinearPairResult(
        rank=down_rank,
        input_features=input_features,
        output_features=output_features
    )

def derive_pair_candidate(
    down_key: str,
    convention: PairConvention,
) -> PairCandidate | None:
    down_suffix = convention["down_suffix"]

    if not down_key.endswith(down_suffix):
        return None

    target = down_key[:-len(down_suffix)]

    if not target:
        raise ValueError(f"Invalid LoRA down key: {down_key}")

    return PairCandidate(
        convention_name=convention["name"],
        target=target,
        down_key=down_key,
        up_key=target + convention["up_suffix"],
    )

def discover_linear_pairs(
    header: SourceModelHeader,
) -> PairDiscoveryResult:
    tensor_names = set(header.tensors)

    pairs: list[DiscoveredLinearPair] = []
    paired_down_keys: set[str] = set()
    paired_up_keys: set[str] = set()
    unpaired_down_keys: set[str] = set()
    unpaired_up_keys: set[str] = set()

    for convention in PAIR_CONVENTIONS:
        for down_key in sorted(tensor_names):
            candidate = derive_pair_candidate(down_key, convention)

            if candidate is None or down_key in paired_down_keys:
                continue

            up_key = candidate["up_key"]

            if up_key not in tensor_names:
                unpaired_down_keys.add(down_key)
                continue

            pair_result = inspect_linear_pair(
                header,
                down_key=down_key,
                up_key=up_key,
            )

            pairs.append(
                DiscoveredLinearPair(
                    convention_name=candidate["convention_name"],
                    target=candidate["target"],
                    down_key=candidate["down_key"],
                    up_key=candidate["up_key"],
                    rank=pair_result["rank"],
                    input_features=pair_result["input_features"],
                    output_features=pair_result["output_features"],
                )
            )

            paired_down_keys.add(down_key)
            paired_up_keys.add(up_key)

    for convention in PAIR_CONVENTIONS:
        down_suffix = convention["down_suffix"]
        up_suffix = convention["up_suffix"]

        for up_key in sorted(tensor_names):
            if up_key in paired_up_keys or not up_key.endswith(up_suffix):
                continue

            target = up_key[:-len(up_suffix)]
            expected_down_key = target + down_suffix

            if expected_down_key not in tensor_names:
                unpaired_up_keys.add(up_key)

    return PairDiscoveryResult(
        pairs=pairs,
        unpaired_down_keys=sorted(unpaired_down_keys),
        unpaired_up_keys=sorted(unpaired_up_keys),
    )

def classify_adapter_tensor_key(
    key: str,
) -> AdapterTensorClassification:
    if key.endswith(".diff"):
        return AdapterTensorClassification(
            kind="additive_tensor_delta",
            target=key[:-len(".diff")],
            contract="additive_tensor_delta",
        )

    if key.endswith(".alpha"):
        return AdapterTensorClassification(
            kind="alpha",
            target=key[:-len(".alpha")],
            contract="alpha",
        )

    for suffix, contract in _UNSUPPORTED_CONTRACT_SUFFIXES:
        if key.endswith(suffix):
            return AdapterTensorClassification(
                kind="unsupported",
                target=key[:-len(suffix)],
                contract=contract,
            )

    return AdapterTensorClassification(
        kind="unsupported",
        target=None,
        contract="unsupported",
    )

def discover_additive_deltas(
    header: SourceModelHeader,
) -> list[DiscoveredAdditiveDelta]:
    deltas: list[DiscoveredAdditiveDelta] = []

    for key in sorted(header.tensors):
        classification = classify_adapter_tensor_key(key)

        if classification["kind"] != "additive_tensor_delta":
            continue

        target = classification["target"]

        if target is None or not target:
            raise ValueError(f"Invalid additive delta key: {key}")

        descriptor = header.tensors[key]

        deltas.append(
            DiscoveredAdditiveDelta(
                key=key,
                target=target,
                shape=descriptor["shape"],
                dtype=descriptor["dtype"],
            )
        )

    return deltas

def _make_adapter_tensor_record(
    header: SourceModelHeader,
    key: str,
    kind: AdapterInventoryKind,
    target: str | None,
    contract: AdapterContract,
) -> AdapterTensorRecord:
    descriptor = header.tensors[key]

    return AdapterTensorRecord(
        key=key,
        shape=descriptor["shape"],
        dtype=descriptor["dtype"],
        kind=kind,
        target=target,
        contract=contract,
    )


def _resolve_contract(
    header: SourceModelHeader,
    key: str,
    contract: AdapterContract,
) -> AdapterContract:
    if contract != "oft_or_boft" or not key.endswith(".oft_blocks"):
        return contract

    rank = len(header.tensors[key]["shape"])

    if rank == 3:
        return "oft"

    if rank == 4:
        return "boft"

    return contract

def inspect_adapter_header(
    header: SourceModelHeader,
) -> AdapterInspectionResult:
    pair_result = discover_linear_pairs(header)
    additive_deltas = discover_additive_deltas(header)

    records_by_key: dict[str, AdapterTensorRecord] = {}

    for pair in pair_result["pairs"]:
        records_by_key[pair["down_key"]] = _make_adapter_tensor_record(
            header,
            pair["down_key"],
            "linear_down",
            pair["target"],
            "linear_lora",
        )
        records_by_key[pair["up_key"]] = _make_adapter_tensor_record(
            header,
            pair["up_key"],
            "linear_up",
            pair["target"],
            "linear_lora",
        )

    for key in pair_result["unpaired_down_keys"]:
        records_by_key[key] = _make_adapter_tensor_record(
            header,
            key,
            "unpaired_down",
            None,
            "linear_lora",
        )

    for key in pair_result["unpaired_up_keys"]:
        records_by_key[key] = _make_adapter_tensor_record(
            header,
            key,
            "unpaired_up",
            None,
            "linear_lora",
        )

    for delta in additive_deltas:
        records_by_key[delta["key"]] = _make_adapter_tensor_record(
            header,
            delta["key"],
            "additive_tensor_delta",
            delta["target"],
            "additive_tensor_delta",
        )

    for key in sorted(header.tensors):
        if key in records_by_key:
            continue

        classification = classify_adapter_tensor_key(key)
        contract = _resolve_contract(
            header,
            key,
            classification["contract"],
        )

        if classification["kind"] == "alpha":
            target = classification["target"]

            if target is None or not target:
                raise ValueError(f"Invalid alpha key: {key}")

            records_by_key[key] = _make_adapter_tensor_record(
                header,
                key,
                "alpha",
                target,
                "alpha",
            )
        else:
            target = classification["target"]

            if contract != "unsupported" and not target:
                raise ValueError(
                    f"Invalid {contract} key: {key}"
                )

            records_by_key[key] = _make_adapter_tensor_record(
                header,
                key,
                "unsupported",
                target,
                contract,
            )

    return AdapterInspectionResult(
        pairs=pair_result["pairs"],
        additive_deltas=additive_deltas,
        tensors=[
            records_by_key[key]
            for key in sorted(records_by_key)
        ],
    )
