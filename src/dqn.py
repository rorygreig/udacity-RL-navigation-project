import torch
import numpy as np
from collections import deque

from src.dqn_agent import Agent


class DQN:
    def __init__(self, env, solve_threshold=15.0):
        self.env = env
        self.solve_threshold = solve_threshold

        self.score_window_length = 100

        # get the default brain
        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]

        env_info = env.reset(train_mode=True)[self.brain_name]

        state = env_info.vector_observations[0]
        self.state_size = len(state)

        self.action_size = brain.vector_action_space_size

        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, seed=0, learning_rate=1e-3)

    def train(self, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.005, eps_decay=0.997):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        print(f"Training DQN with {n_episodes} episodes")

        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_length)  # window of scores
        eps = eps_start

        checkpoint_interval = 200

        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            mean_recent_score = np.mean(scores_window)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_recent_score), end="")
            if i_episode % checkpoint_interval == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_recent_score))
                self.store_weights(f"checkpoint_{i_episode}.pth")
            if np.mean(scores_window) >= self.solve_threshold:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                      .format(i_episode - self.score_window_length, mean_recent_score))
                break

        self.env.close()

        return scores

    def store_weights(self, filename='checkpoint.pth'):
        print("Storing weights")
        torch.save(self.agent.qnetwork_local.state_dict(), "weights/" + filename)

    def run_with_stored_weights(self, filename='"final_weights.pth"'):
        # load stored weights from training
        self.agent.qnetwork_local.load_state_dict(torch.load("weights/" + filename))

        print("Running agent with trained weights")
        # run environment with trained agent
        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = self.agent.act(state)  # select greedy action
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("Score: {}".format(score))
