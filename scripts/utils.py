"""
Check that your code works with these visualisers
Do NOT modify this script,
your outputs should match the input types of these visualisers to pass our tests
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

def draw_utility(
        output_path,
        utility,
        grid_np,
        init_pos
        ):

    assert init_pos[0] >= 0 and init_pos[1] >= 0 and init_pos[0] < grid_np.shape[0] and init_pos[1] < grid_np.shape[1] 

    plt.imshow(grid_np, cmap='Greys', vmin=0, vmax=1)

    plt.imshow(utility, cmap='viridis', vmin=0, vmax=1)

    start_mask = np.zeros(grid_np.shape)
    start_mask[init_pos[0], init_pos[1]]=1

    start_masked = np.ma.masked_where(start_mask == 0, start_mask)
    plt.imshow(start_masked, cmap='summer')

    
    for i in range(utility.shape[0]):
        for j in range(utility.shape[1]):
            plt.text( 
                    j, i,
                    f"{utility[i,j]:.2f}",
                    ha = 'center',
                    va = 'center'
                    )
    plt.grid(True, color='lightgray', linewidth=0.5)
    plt.xticks(np.arange(0.5, utility.shape[1], 1))
    plt.yticks(np.arange(0.5, utility.shape[0], 1))
    plt.savefig(output_path)
    plt.close()

def draw_policy(
        output_path,
        policy,
        grid_np,
        ):
    plt.imshow(policy, cmap='viridis', vmin=0, vmax=4)
    policy = policy.astype(int)

    mapper = {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→',
        4: '',
        5: '',
    }

    start_masked = np.ma.masked_where(grid_np == 0, grid_np)
    plt.imshow(start_masked, cmap='Greys', vmin=0, vmax=1)
 
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            plt.text( 
                    j, i,
                    f"{mapper[policy[i,j]]}",
                    ha = 'center',
                    va = 'center'
                    )
    plt.grid(True, color='lightgray', linewidth=0.5)
    plt.xticks(np.arange(0.5, policy.shape[1], 1))
    plt.yticks(np.arange(0.5, policy.shape[0], 1))
    plt.savefig(output_path)
    plt.close()
