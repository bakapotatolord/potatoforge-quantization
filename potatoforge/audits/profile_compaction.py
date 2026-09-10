"""Compact explicit tensor actions into ordered runtime profile rules."""

from collections.abc import Mapping

from ..profiles import ProfileRule, QuantizationAction


def _grouping_candidates(tensor_name: str) -> tuple[tuple[str, str], ...]:
    final_dot = tensor_name.rfind(".")
    second_last_dot = tensor_name.rfind(".", 0, final_dot)
    if second_last_dot < 0:
        return ()
    suffix = tensor_name[second_last_dot + 1 :]
    return tuple(
        (tensor_name[:index + 1], suffix)
        for index, character in enumerate(tensor_name[:second_last_dot + 1])
        if character == "."
    )


def _exact_rule(
    action: QuantizationAction,
    tensor_name: str,
) -> ProfileRule:
    return {
        "action": action,
        "prefix": tensor_name,
        "suffixes": ("",),
    }


def compact_runtime_rules(
    desired_actions: Mapping[str, QuantizationAction],
    default_action: QuantizationAction,
) -> tuple[ProfileRule, ...]:
    """Return first-match-wins rules equivalent to the desired actions."""
    candidate_suffixes: dict[tuple[QuantizationAction, str], set[str]] = {}
    for tensor_name, action in desired_actions.items():
        if action == default_action:
            continue
        for prefix, suffix in _grouping_candidates(tensor_name):
            candidate_suffixes.setdefault((action, prefix), set()).add(suffix)

    candidates: list[
        tuple[
            QuantizationAction,
            str,
            tuple[str, ...],
            set[str],
            set[str],
        ]
    ] = []
    for (action, prefix), raw_suffixes in candidate_suffixes.items():
        suffixes = tuple(sorted(raw_suffixes))
        matched_names = {
            tensor_name
            for tensor_name in desired_actions
            if tensor_name.startswith(prefix)
            and tensor_name.endswith(suffixes)
        }
        target_names = {
            tensor_name
            for tensor_name in matched_names
            if desired_actions[tensor_name] == action
        }
        if len(target_names) > 1:
            candidates.append(
                (action, prefix, suffixes, target_names, matched_names)
            )

    # Rule resolution is first-match-wins, so retain deepest-prefix-first order.
    candidates.sort(
        key=lambda item: (-len(item[1]), item[0], item[1], item[2])
    )
    covered: set[str] = set()
    exceptions: dict[str, QuantizationAction] = {}
    grouped_rules: list[ProfileRule] = []
    for action, prefix, suffixes, target_names, matched_names in candidates:
        uncovered_names = target_names - covered
        if len(uncovered_names) < 2:
            continue
        mismatches = matched_names - target_names
        new_exceptions = mismatches - covered - set(exceptions)
        if len(uncovered_names) <= 1 + len(new_exceptions):
            continue
        grouped_rules.append(
            {
                "action": action,
                "prefix": prefix,
                "suffixes": suffixes,
            }
        )
        covered.update(target_names)
        exceptions.update(
            {
                tensor_name: desired_actions[tensor_name]
                for tensor_name in new_exceptions
            }
        )

    exception_rules = [
        _exact_rule(action, tensor_name)
        for tensor_name, action in sorted(exceptions.items())
    ]
    exact_rules = [
        _exact_rule(desired_actions[tensor_name], tensor_name)
        for tensor_name in sorted(desired_actions)
        if desired_actions[tensor_name] != default_action
        and tensor_name not in covered
        and tensor_name not in exceptions
    ]
    return tuple(exception_rules + grouped_rules + exact_rules)
