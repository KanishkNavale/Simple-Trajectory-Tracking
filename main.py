# Library Imports
import numpy as np
import gym
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
 
# Main Routine
if __name__ == '__main__':
    
    # Load the environment
    env = gym.make('FetchReach-v1')
    alpha = 2

    # Initiate the environment
    state = env.reset().values()
    pure_states, curr_pos, goal_pos = state
    
    # Run a episode
    y0 = curr_pos
    poserr_log = []
    mean_score=0
    for i in range(1, env._max_episode_steps+1):
        
        # Render the Environment
        env.render()
        
        # Calculate the Trajectory
        y = curr_pos
        y_cap = y0 + (((alpha * i)/(env._max_episode_steps+1))*(goal_pos - y0))
        action = (y_cap - y)
        action = np.append(action, 0)
        
        # Take a step using computed action    
        next_state, reward, done, info =  env.step(action)
        _pure_states, _curr_pos, _goal_pos = next_state.values()
        
        # Print the Positional Error
        pos_err = np.abs(np.mean(goal_pos - _curr_pos))
        print (f'Positional Error is {pos_err:.4f} at Step {i}')
        
        curr_pos = _curr_pos

        poserr_log.append(pos_err)
        mean_score += reward
        
    # Close the Environment
    env.close()
    print (f'Mean Episodic Reward: {mean_score}')
    
    plt.plot(poserr_log, c='red')
    plt.xlabel('Step')
    plt.ylabel('Positional Error')
    plt.grid(True)
    plt.savefig('PosError.png')