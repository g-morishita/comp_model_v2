"""Stan data builders for event-based subject and dataset structures.

The Stan exporter is a derived view over the canonical event hierarchy. It does
not define the primary ontology; it flattens extracted decision views into the
array layout expected by the Stan programs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comp_model.data.extractors import extract_decision_views

if TYPE_CHECKING:
    from collections.abc import Iterable

    from comp_model.data.extractors import DecisionTrialView
    from comp_model.data.schema import Dataset, SubjectData
    from comp_model.models.condition.shared_delta import SharedDeltaLayout
    from comp_model.models.kernels.base import ModelKernelSpec
    from comp_model.tasks.schemas import TrialSchema


def subject_to_stan_data(subject_data: SubjectData, schema: TrialSchema) -> dict[str, Any]:
    """Export one subject's data to Stan's flat array format.

    Parameters
    ----------
    subject_data
        Subject data to flatten.
    schema
        Trial schema used to extract decision views.

    Returns
    -------
    dict[str, Any]
        Stan-ready flat arrays with contiguous 1-based action indexing.

    Notes
    -----
    The exporter iterates over blocks and trials in observed order, extracts
    :class:`~comp_model.data.extractors.DecisionTrialView` records, remaps any
    raw action identifiers to contiguous 1-based Stan indices, and emits arrays
    aligned trial-by-trial for choice, reward, availability mask, block index,
    and optional social information.
    """

    trials_flat: list[DecisionTrialView] = []
    block_of_trial: list[int] = []

    for block in subject_data.blocks:
        for trial in block.trials:
            for view in extract_decision_views(trial, schema):
                trials_flat.append(view)
                block_of_trial.append(block.block_index + 1)

    if not trials_flat:
        raise ValueError("No decision trials found in subject data")

    action_to_index = _action_index_mapping(trials_flat)
    n_actions = len(action_to_index)
    total_trials = len(trials_flat)

    choice = [0] * total_trials
    reward = [0.0] * total_trials
    avail_mask: list[list[float]] = [[0.0] * n_actions for _ in range(total_trials)]
    social_action = [0] * total_trials
    social_reward = [0.0] * total_trials
    has_social = [0] * total_trials

    for index, view in enumerate(trials_flat):
        choice[index] = action_to_index[view.choice]
        reward[index] = float(view.reward) if view.reward is not None else 0.0
        for action in view.available_actions:
            avail_mask[index][action_to_index[action] - 1] = 1.0
        if view.social_action is not None:
            has_social[index] = 1
            social_action[index] = action_to_index[view.social_action]
            social_reward[index] = (
                float(view.social_reward) if view.social_reward is not None else 0.0
            )

    return {
        "A": n_actions,
        "T": total_trials,
        "choice": choice,
        "reward": reward,
        "avail_mask": avail_mask,
        "block_of_trial": block_of_trial,
        "social_action": social_action,
        "social_reward": social_reward,
        "has_social": has_social,
    }


def dataset_to_stan_data(dataset: Dataset, schema: TrialSchema) -> dict[str, Any]:
    """Export a multi-subject dataset for hierarchical Stan models.

    Parameters
    ----------
    dataset
        Dataset to flatten.
    schema
        Trial schema used to extract decision views.

    Returns
    -------
    dict[str, Any]
        Stan-ready hierarchical arrays with subject indexing.

    Notes
    -----
    Dataset export is implemented by first exporting each subject independently
    with :func:`subject_to_stan_data`, then concatenating those subject chunks in
    subject-major order while adding a 1-based ``subj`` index.
    """

    subject_chunks = [subject_to_stan_data(subject, schema) for subject in dataset.subjects]
    action_counts = {chunk["A"] for chunk in subject_chunks}
    if len(action_counts) != 1:
        raise ValueError("All subjects must have the same number of actions")
    n_actions = action_counts.pop()

    all_choice: list[int] = []
    all_reward: list[float] = []
    all_avail_mask: list[list[float]] = []
    all_subject: list[int] = []
    all_block_of_trial: list[int] = []
    all_social_action: list[int] = []
    all_social_reward: list[float] = []
    all_has_social: list[int] = []

    for subject_index, chunk in enumerate(subject_chunks, start=1):
        for trial_index in range(chunk["T"]):
            all_choice.append(chunk["choice"][trial_index])
            all_reward.append(chunk["reward"][trial_index])
            all_avail_mask.append(chunk["avail_mask"][trial_index])
            all_subject.append(subject_index)
            all_block_of_trial.append(chunk["block_of_trial"][trial_index])
            all_social_action.append(chunk["social_action"][trial_index])
            all_social_reward.append(chunk["social_reward"][trial_index])
            all_has_social.append(chunk["has_social"][trial_index])

    return {
        "N": len(dataset.subjects),
        "A": n_actions,
        "T": len(all_choice),
        "choice": all_choice,
        "reward": all_reward,
        "avail_mask": all_avail_mask,
        "subj": all_subject,
        "block_of_trial": all_block_of_trial,
        "social_action": all_social_action,
        "social_reward": all_social_reward,
        "has_social": all_has_social,
    }


def add_condition_data(
    stan_data: dict[str, Any],
    subject_data: SubjectData,
    layout: SharedDeltaLayout,
) -> None:
    """Add within-subject condition indices to a Stan data dictionary.

    Parameters
    ----------
    stan_data
        Stan data dictionary to mutate.
    subject_data
        Subject data used to derive condition indices.
    layout
        Shared-plus-delta layout defining condition order.

    Returns
    -------
    None
        This function mutates ``stan_data`` in-place.

    Notes
    -----
    Condition IDs are assigned in ``layout.conditions`` order. The baseline
    condition is exported separately so Stan programs can reconstruct
    shared-plus-delta parameterizations consistently with Python.
    """

    condition_to_id = {
        condition: index for index, condition in enumerate(layout.conditions, start=1)
    }
    condition_index: list[int] = []
    for block in subject_data.blocks:
        if block.condition not in condition_to_id:
            raise ValueError(f"Unknown condition {block.condition!r}")
        condition_id = condition_to_id[block.condition]
        for _ in block.trials:
            condition_index.append(condition_id)

    stan_data["C"] = len(layout.conditions)
    stan_data["baseline_cond"] = condition_to_id[layout.baseline_condition]
    stan_data["cond"] = condition_index


def add_condition_data_dataset(
    stan_data: dict[str, Any],
    dataset: Dataset,
    layout: SharedDeltaLayout,
) -> None:
    """Add within-subject condition indices for a multi-subject dataset.

    Parameters
    ----------
    stan_data
        Stan data dictionary to mutate.
    dataset
        Dataset whose subjects provide per-block condition labels.
    layout
        Shared-plus-delta layout defining condition order.

    Returns
    -------
    None
        This function mutates ``stan_data`` in-place.
    """

    condition_to_id = {
        condition: index for index, condition in enumerate(layout.conditions, start=1)
    }
    condition_index: list[int] = []
    for subject in dataset.subjects:
        for block in subject.blocks:
            if block.condition not in condition_to_id:
                raise ValueError(f"Unknown condition {block.condition!r}")
            condition_id = condition_to_id[block.condition]
            for _ in block.trials:
                condition_index.append(condition_id)

    stan_data["C"] = len(layout.conditions)
    stan_data["baseline_cond"] = condition_to_id[layout.baseline_condition]
    stan_data["cond"] = condition_index


def add_prior_data(stan_data: dict[str, Any], kernel_spec: ModelKernelSpec) -> None:
    """Add prior hyperparameters from kernel metadata to a Stan data dictionary.

    Parameters
    ----------
    stan_data
        Stan data dictionary to mutate.
    kernel_spec
        Kernel specification whose parameter priors should be exported.

    Returns
    -------
    None
        This function mutates ``stan_data`` in-place.

    Notes
    -----
    Prior means and scales are exported on the unconstrained parameterization
    used by the kernel specification. Parameters without an explicit prior fall
    back to a weak ``Normal(0, 2)`` convention.
    """

    for parameter in kernel_spec.parameter_specs:
        if parameter.prior is not None:
            mu = parameter.prior.kwargs.get("mu", 0.0)
            sigma = parameter.prior.kwargs.get("sigma", 2.0)
        else:
            mu = 0.0
            sigma = 2.0
        stan_data[f"{parameter.name}_prior_mu"] = float(mu)
        stan_data[f"{parameter.name}_prior_sigma"] = float(sigma)


def add_state_reset_data(stan_data: dict[str, Any], kernel_spec: ModelKernelSpec) -> None:
    """Add kernel state-reset metadata to a Stan data dictionary.

    Parameters
    ----------
    stan_data
        Stan data dictionary to mutate.
    kernel_spec
        Kernel specification whose reset policy should be exported.

    Returns
    -------
    None
        This function mutates ``stan_data`` in-place.

    Raises
    ------
    ValueError
        Raised when the kernel exposes an unknown reset policy.

    Notes
    -----
    Stan receives the reset policy as an integer flag because the compiled
    program cannot inspect Python-side kernel metadata directly.
    """

    if kernel_spec.state_reset_policy == "per_subject":
        stan_data["reset_on_block"] = 0
        return
    if kernel_spec.state_reset_policy == "per_block":
        stan_data["reset_on_block"] = 1
        return
    raise ValueError(f"Unknown state_reset_policy: {kernel_spec.state_reset_policy!r}")


def _action_index_mapping(trials_flat: Iterable[DecisionTrialView]) -> dict[int, int]:
    """Build a contiguous Stan action index for observed action identifiers.

    Parameters
    ----------
    trials_flat
        Decision views whose action identifiers should be encoded.

    Returns
    -------
    dict[int, int]
        Mapping from raw action identifier to 1-based Stan index.

    Notes
    -----
    Actions are assigned indices in first-seen order across available actions,
    realized subject choices, and any observed social actions. This keeps Stan
    arrays compact even when raw action IDs are sparse.
    """

    ordered_actions: list[int] = []
    seen_actions: set[int] = set()

    for view in trials_flat:
        for action in view.available_actions:
            if action not in seen_actions:
                seen_actions.add(action)
                ordered_actions.append(action)
        if view.choice not in seen_actions:
            seen_actions.add(view.choice)
            ordered_actions.append(view.choice)
        if view.social_action is not None and view.social_action not in seen_actions:
            seen_actions.add(view.social_action)
            ordered_actions.append(view.social_action)

    return {action: index for index, action in enumerate(ordered_actions, start=1)}
