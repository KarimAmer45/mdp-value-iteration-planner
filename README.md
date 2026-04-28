# MDP Value Iteration Planner

Value iteration implementation for a grid-world Markov decision process with stochastic movement.

## Contents

- `scripts/assignment_2_mdp.py` implements:
  - safe grid transitions with obstacle and boundary reflection
  - Bellman value iteration
  - policy extraction with the required action encoding
- `scripts/example_test_case.py` draws utility and policy plots for a smoke test.
- `scripts/utils.py` contains the provided plotting helpers.
- `mcq.txt` contains the multiple-choice answers in the required format.

## Model

- Free cells are grid values below `0.5`; occupied cells are `0.5` or higher.
- Actions follow the assignment convention: `0` up, `1` down, `2` left, `3` right, `4` stay.
- Moving succeeds with probability `p`; failure probability is split equally across the other three movement directions.
- The stay action is deterministic.
- Terminal-state utilities are fixed to their reward values.

## Run

```bash
cd scripts
python example_test_case.py
```

The example script writes utility and policy plots for the test case.
