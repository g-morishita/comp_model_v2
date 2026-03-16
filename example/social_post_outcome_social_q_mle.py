"""Simulate social Q-learning agents with post-outcome demonstrator observation,
then recover parameters with per-subject MLE.

Ground-truth: alpha_self=0.3, alpha_other=0.2, beta=2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from comp_model.data import Dataset
from comp_model.data.schema import Event, EventPhase
from comp_model.environments import StationaryBanditEnvironment
from comp_model.inference import fit
from comp_model.inference.config import HierarchyStructure, InferenceConfig
from comp_model.inference.mle.optimize import MleOptimizerConfig
from comp_model.io import load_dataset_from_csv, save_dataset_to_csv
from comp_model.models.kernels import SocialObservedOutcomeQKernel, SocialQParams
from comp_model.runtime import SimulationConfig, simulate_dataset
from comp_model.tasks import SOCIAL_POST_OUTCOME_SCHEMA, BlockSpec, TaskSpec


# ── Social environment wrapper ──────────────────────────────────────────────
@dataclass(slots=True)
class SocialBanditEnvironment:
    """Wraps StationaryBanditEnvironment to inject demonstrator observations."""

    inner: StationaryBanditEnvironment
    demo_reward_probs: tuple[float, ...] = ()
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    @property
    def environment_id(self) -> str:
        return "social_bandit"

    def reset(self, block_spec: BlockSpec, *, rng: np.random.Generator) -> None:
        self._rng = rng
        self.inner.reset(block_spec, rng=rng)

    def step(self, action: int | None = None) -> tuple[Event, ...]:
        assert self.inner._block_spec is not None
        schema = self.inner._block_spec.schema
        schema_step = schema.steps[self.inner._step_index]

        if schema_step.phase == EventPhase.INPUT and schema_step.actor_id != "subject":
            assert self._rng is not None
            n_actions = self.inner.n_actions
            demo_action = int(self._rng.integers(0, n_actions))
            probs = self.demo_reward_probs if self.demo_reward_probs else self.inner.reward_probs
            demo_reward = float(self._rng.random() < probs[demo_action])

            events = self.inner.step(action=None)
            original = events[0]
            patched = Event(
                phase=original.phase,
                event_index=original.event_index,
                node_id=original.node_id,
                actor_id=original.actor_id,
                payload={
                    "available_actions": original.payload["available_actions"],
                    "observation": {
                        "social_action": demo_action,
                        "social_reward": demo_reward,
                    },
                },
            )
            return (patched,)

        return self.inner.step(action=action)


# ── 1. Define task ──────────────────────────────────────────────────────────
N_ACTIONS = 2
N_TRIALS = 100
N_SUBJECTS = 5

task = TaskSpec(
    task_id="social_post_outcome",
    blocks=(
        BlockSpec(
            condition="social",
            n_trials=N_TRIALS,
            schema=SOCIAL_POST_OUTCOME_SCHEMA,
            metadata={"n_actions": N_ACTIONS},
        ),
    ),
)

# ── 2. Ground-truth parameters ─────────────────────────────────────────────
TRUE_ALPHA_SELF = 0.3
TRUE_ALPHA_OTHER = 0.2
TRUE_BETA = 2.0
REWARD_PROBS = (0.8, 0.2)

kernel = SocialObservedOutcomeQKernel()
true_params = SocialQParams(
    alpha_self=TRUE_ALPHA_SELF, alpha_other=TRUE_ALPHA_OTHER, beta=TRUE_BETA
)

# ── 3. Simulate dataset ────────────────────────────────────────────────────
params_per_subject = {f"sub_{i:02d}": true_params for i in range(N_SUBJECTS)}

dataset = simulate_dataset(
    task=task,
    env_factory=lambda: SocialBanditEnvironment(
        inner=StationaryBanditEnvironment(n_actions=N_ACTIONS, reward_probs=REWARD_PROBS),
    ),
    kernel=kernel,
    params_per_subject=params_per_subject,
    config=SimulationConfig(seed=42),
)

# ── 4. Save and reload CSV ─────────────────────────────────────────────────
csv_path = Path(__file__).parent / "social_post_outcome_data.csv"
save_dataset_to_csv(dataset, schema=SOCIAL_POST_OUTCOME_SCHEMA, path=csv_path)
print(f"Saved {len(dataset.subjects)} subjects to {csv_path}")

loaded = load_dataset_from_csv(csv_path, schema=SOCIAL_POST_OUTCOME_SCHEMA)

# ── 5. Fit each subject with MLE ───────────────────────────────────────────
mle_config = InferenceConfig(
    hierarchy=HierarchyStructure.SUBJECT_SHARED,
    backend="mle",
    mle_config=MleOptimizerConfig(n_restarts=10, seed=0),
)

print(
    f"\n{'Subject':<12} {'True αs':>8} {'Fit αs':>8} "
    f"{'True αo':>8} {'Fit αo':>8} "
    f"{'True β':>8} {'Fit β':>8} {'LL':>10}"
)
print("-" * 80)

for subject in loaded.subjects:
    result = fit(mle_config, kernel, subject, SOCIAL_POST_OUTCOME_SCHEMA)
    cp = result.constrained_params
    print(
        f"{subject.subject_id:<12} "
        f"{TRUE_ALPHA_SELF:>8.3f} {cp['alpha_self']:>8.3f} "
        f"{TRUE_ALPHA_OTHER:>8.3f} {cp['alpha_other']:>8.3f} "
        f"{TRUE_BETA:>8.3f} {cp['beta']:>8.3f} "
        f"{result.log_likelihood:>10.2f}"
    )

print("\nDone.")
