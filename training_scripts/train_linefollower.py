#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.95
    epsilon_decay = 0.99

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")
                saveQ(q_table, "q_table.txt")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    best_next_q = np.max(q_table[next_state])
    td_target = reward + gamma * best_next_q
    q_table[state, action] += alpha * (td_target - q_table[state, action])

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    if random.random() < epsilon:
        # Explore pick a completely random action
        action = action_space.sample()
    else:
        # Exploit
        action = int(np.argmax(q_table[state]))

    return action


def saveQ(q_table, save_path):
    # Plain text: one row per state, one column per action
    np.savetxt(save_path, q_table, fmt="%.6f",
               header="rows: states (left, center, right) | "
                      "columns: actions (left, forward, right)")
    print(f"Saved Q table to {save_path}")

def loadQ(load_path):
    # loadtxt skips the "#" header line automatically
    q_table = np.loadtxt(load_path)
    print(f"Loaded Q table from {load_path}")
    return q_table

if __name__ == "__main__":
    main()