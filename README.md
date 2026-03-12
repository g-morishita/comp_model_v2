# comp_model

Computational modeling toolkit with event-based task semantics.

Run the concrete simulation-and-fitting example from the repository root with
`uv run python -m example.asocial_q_learning`.

The example builds a bandit `TaskSpec`, simulates a subject with
`AsocialQLearningKernel`, and recovers parameters with the public MLE dispatch
entry point.
