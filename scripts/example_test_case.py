"""Script to run and visualise the outputs
Use this script to check that your code works.
You can uncomment the code based on which function you want to test.
I highly recommend you write your own test functions as well.

BIG HINT: we use a very similar script to evaluate your code.
So make sure that your code works well in this script!
"""

import os 

import numpy as np 

import assignment_2_mdp as mdp 
from utils import draw_utility, draw_policy

def test_1(output_dir):
    grid_np = np.zeros((10,10))
    reward_np = np.zeros(grid_np.shape)
    reward_np[9,9]=1
    terminal_states = [(9,9)]
    init_pos = (0,0)
    vi = mdp.VI(
            states_grid = grid_np,
            transition_p=0.75,
            rewards = reward_np,
            init_pos = init_pos,
            terminal_states = terminal_states,
            gamma = 0.9
            )
    utility = vi.calculate_utility()
    draw_utility(
            os.path.join(output_dir, "test1.png"),
            utility,
            grid_np,
            init_pos
            )
    policy = vi.get_policy()
    draw_policy(
            os.path.join(output_dir, "test1_policy.png"),
            policy,
            grid_np,
            )

def test_2(output_dir):
    # TODO make your own tests
    # make sure to check edge cases
    pass

if __name__ == '__main__':
    output_dir = ""

    print('Test 1:')
    test_1(output_dir)
    
