"""Script containing the MDP class

You task is to modify all the classes in this script to solve the MDP
You are free to create your own classes to aid this process as you like
"""

import numpy as np 
from typing import Tuple, List  # you may need to change this depending on your python version


ACTIONS = (
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (0, 0),   # stay
        )

class VI:
    """Class for the value iterative algorithm
    """
    def __init__(
            self,
            states_grid: np.ndarray,
            transition_p: float,
            rewards: np.ndarray,
            init_pos: Tuple[int, int],
            terminal_states: List[Tuple[int, int]],
            gamma=1.0,
            max_iterations=1000,
            eps = 10**-10
            ):
        self.states_grid = states_grid
        self.transition_p = transition_p
        self.rewards = rewards.astype(float, copy=False)
        self.init_pos = init_pos
        self.terminal_states = set(tuple(s) for s in terminal_states)
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.eps = eps
        self.transition_model = StateTransitionModel(transition_p)
        self.values = np.zeros(self.states_grid.shape, dtype=float)

    def is_unoccupied(self, i, j):
        return (
                0 <= i < self.states_grid.shape[0]
                and 0 <= j < self.states_grid.shape[1]
                and self.states_grid[i, j] < 0.5
                )

    def move_safely(self, i, j, di ,dj):
        """
        Move inside free space only. Attempts to leave the grid or enter an
        occupied cell reflect back into the original state.
        """
        new_i = i + di
        new_j = j + dj
        if not self.is_unoccupied(new_i, new_j):
            return i, j
        return new_i, new_j

    def _expected_value(self, i, j, action, values):
        if action == 4:
            return values[i, j]

        expected = 0.0
        for other_action, (di, dj) in enumerate(ACTIONS[:4]):
            prob = (
                    self.transition_model.success_prob
                    if other_action == action
                    else self.transition_model.failure_prob
                    )
            ni, nj = self.move_safely(i, j, di, dj)
            expected += prob * values[ni, nj]
        return expected

    def calculate_utility(self):
        """
        returns:
            numpy array of size states_grid map with the utility at each unoccupied state
            The occupied cells or the terminal states can have arbitrary (float) values.
        """
        values = np.zeros(self.states_grid.shape, dtype=float)
        for terminal in self.terminal_states:
            values[terminal] = self.rewards[terminal]

        for _ in range(self.max_iterations):
            updated = values.copy()
            max_delta = 0.0

            for i in range(self.states_grid.shape[0]):
                for j in range(self.states_grid.shape[1]):
                    if not self.is_unoccupied(i, j):
                        updated[i, j] = 0.0
                        continue
                    if (i, j) in self.terminal_states:
                        updated[i, j] = self.rewards[i, j]
                        continue

                    best_future = max(
                            self._expected_value(i, j, action, values)
                            for action in range(5)
                            )
                    updated[i, j] = self.rewards[i, j] + self.gamma * best_future
                    max_delta = max(max_delta, abs(updated[i, j] - values[i, j]))

            values = updated
            if max_delta < self.eps:
                break

        self.values = values
        return values

    def get_policy(self):
        """Assumes that you have ran calculate_utility to get the utility in some instance attribute
        returns:
            numpy array of size states_grid map with integers encoding the action
            follow the convention 0: up, 1: down, 2: left, 3: right, 4: stay still. 
            The occupied cells or the terminal states can have arbitrary (integer) values.
        """
        values = self.values
        policy = np.full(self.states_grid.shape, 4, dtype=int)

        for i in range(self.states_grid.shape[0]):
            for j in range(self.states_grid.shape[1]):
                if not self.is_unoccupied(i, j) or (i, j) in self.terminal_states:
                    continue

                action_values = [
                        self._expected_value(i, j, action, values)
                        for action in range(5)
                        ]
                policy[i, j] = int(np.argmax(action_values))

        return policy 

class StateTransitionModel:
    """model transition from state s to s' with action a
    """
    def __init__(
            self, 
            success_prob
            ):
        self.success_prob = success_prob
        self.failure_prob = (1 - success_prob) / 3
