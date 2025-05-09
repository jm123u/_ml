import gymnasium as gym
import numpy as np

def optimized_balanced_policy(observation):
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    
    
    if abs(pole_angle) < 0.03 and abs(pole_velocity) < 0.1:
        k_pos = 1.2  
        k_vel = 0.6  
        k_angle = 2.5  
        k_ang_vel = 1.0 

    elif abs(pole_angle) > 0.1:
        k_pos = 0.8  
        k_vel = 0.3  
        k_angle = 4.5  
        k_ang_vel = 1.5  

    else:
        k_pos = 1.0
        k_vel = 0.5
        k_angle = 3.5
        k_ang_vel = 1.2
    
    control = (
        k_pos * cart_position +        
        k_vel * cart_velocity +        
        k_angle * pole_angle +         
        k_ang_vel * pole_velocity      
    )
    
    if abs(cart_position) > 1.5:
        position_correction = 1.5 * np.sign(cart_position)
        control += position_correction
    

    if abs(control) < 0.1:
        
        return 0 if np.random.random() < 0.5 + control * 3 else 1
    else:
        
        return 0 if control < 0 else 1


env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
steps = 0
episode_count = 0
total_steps = 0
max_episode_steps = 0

for _ in range(5000):  
    env.render()
    action = optimized_balanced_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1
    
    if terminated or truncated:
        print(f"Episode {episode_count + 1}: {steps} steps")
        max_episode_steps = max(max_episode_steps, steps)
        episode_count += 1
        total_steps += steps
        observation, info = env.reset()
        steps = 0

env.close()
