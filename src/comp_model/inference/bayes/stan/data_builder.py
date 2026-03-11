"""Stan data builders for event-based subject and dataset structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comp_model.data.extractors import extract_decision_views

if TYPE_CHECKING:
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
        Stan-ready flat arrays with 1-based action indexing.
    """

    trials_flat: list[DecisionTrialView] = []
    block_of_trial: list[int] = []
    block_starts: list[int] = []
    n_trials_in_block: list[int] = []

    trial_counter = 0
    for block in subject_data.blocks:
        block_starts.append(trial_counter + 1)
        block_trial_count = 0
        for trial in block.trials:
            for view in extract_decision_views(trial, schema):
                trials_flat.append(view)
                block_of_trial.append(block.block_index + 1)
                trial_counter += 1
                block_trial_count += 1
        n_trials_in_block.append(block_trial_count)

    if not trials_flat:
        raise ValueError("No decision trials found in subject data")

    action_counts: set[int] = {max(view.available_actions) + 1 for view in trials_flat}
    if len(action_counts) != 1:
        raise ValueError("All decision views must agree on the number of actions")
    n_actions = action_counts.pop()
    total_trials = len(trials_flat)

    choice = [0] * total_trials
    reward = [0.0] * total_trials
    avail_mask: list[list[float]] = [[0.0] * n_actions for _ in range(total_trials)]
    social_action = [0] * total_trials
    social_reward = [0.0] * total_trials
    has_social = [0] * total_trials

    for index, view in enumerate(trials_flat):
        choice[index] = view.choice + 1
        reward[index] = float(view.reward) if view.reward is not None else 0.0
        for action in view.available_actions:
            avail_mask[index][action] = 1.0
        if view.social_action is not None:
            has_social[index] = 1
            social_action[index] = view.social_action + 1
            social_reward[index] = (
                float(view.social_reward) if view.social_reward is not None else 0.0
            )

    return {
        "A": n_actions,
        "T": total_trials,
        "B": len(block_starts),
        "choice": choice,
        "reward": reward,
        "avail_mask": avail_mask,
        "block_of_trial": block_of_trial,
        "block_start": block_starts,
        "n_trials_in_block": n_trials_in_block,
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
