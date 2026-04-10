"""Regression tests for demo-only sticky Stan program source."""

from __future__ import annotations

import re
from pathlib import Path

from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_bias_sticky import (
    SocialRlDemoActionBiasStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_action_sticky import (
    SocialRlDemoActionStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_mixture_sticky import (
    SocialRlDemoMixtureStickyStanAdapter,
)
from comp_model.inference.bayes.stan.adapters.social_rl_demo_reward_sticky import (
    SocialRlDemoRewardStickyStanAdapter,
)
from comp_model.inference.config import HierarchyStructure

_BUGGY_UPDATE_RE = re.compile(r"last_self_choice(?:\[n\])?\s*=\s*step_update_action\[e\];")
_CHOICE_UPDATE_RE = re.compile(r"last_self_choice(?:\[n\])?\s*=\s*step_choice\[e\];")


def test_demo_only_sticky_stan_programs_update_self_choice_on_decision_rows() -> None:
    """Demo-only sticky Stan programs must refresh stickiness state on choices."""

    adapters = (
        SocialRlDemoActionStickyStanAdapter(),
        SocialRlDemoActionBiasStickyStanAdapter(),
        SocialRlDemoRewardStickyStanAdapter(),
        SocialRlDemoMixtureStickyStanAdapter(),
    )

    hierarchies = (
        HierarchyStructure.SUBJECT_SHARED,
        HierarchyStructure.STUDY_SUBJECT,
        HierarchyStructure.SUBJECT_BLOCK_CONDITION,
        HierarchyStructure.STUDY_SUBJECT_BLOCK_CONDITION,
    )

    for adapter in adapters:
        for hierarchy in hierarchies:
            source = Path(adapter.stan_program_path(hierarchy)).read_text(encoding="utf-8")
            assert _CHOICE_UPDATE_RE.search(source), (
                f"{adapter.kernel_spec().model_id} / {hierarchy} must update "
                "last_self_choice on step_choice rows."
            )
            assert _BUGGY_UPDATE_RE.search(source) is None, (
                f"{adapter.kernel_spec().model_id} / {hierarchy} still updates "
                "last_self_choice from step_update_action."
            )
