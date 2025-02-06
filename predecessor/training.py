# load library
import copy
import numpy as np
from tqdm import tqdm

# load custom modules
from predecessor.env import compute_true_q_value


class training_loop:
    def __init__(
        self, env, agent, agent_type: str = "Q", episodes=500, max_step_length=100
    ):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_step_length = max_step_length

        self.step_lengths = []
        self.rewards = []
        self.Q_error_history = []
        self.agent_type = agent_type
        self.cumulative_reward = 0

        self.agent.reset()
        # compute true q values works only for my custom 1D env
        # should be avoided for other envs like neuro-nav
        self.true_q_values = compute_true_q_value(self.env, gamma=self.agent.gamma)

    def compute_q_error(self):
        estimated_q_values = self.agent.q_table
        q_error = self.true_q_values - estimated_q_values
        mse_q_error = np.mean(np.square(q_error[:, :-1]))
        return mse_q_error

    def run(self):
        for i in range(self.episodes):
            current_state = self.env.reset()
            done = False
            # reward_error = []

            # step_idx = 0 for while loop
            # while True:
            for step_idx in range(self.max_step_length):
                action = self.agent.sample_action(current_state)
                next_state, reward, done, _ = self.env.step(action)
                current_exp = (current_state, action, reward, next_state, done)
                if self.agent_type == "Q":
                    self.agent.update_q_weights(current_exp)
                elif self.agent_type == "SF":
                    self.agent.update_w(current_exp)
                    self.agent.update_r(current_exp)
                elif self.agent_type == "PF":
                    self.agent.input(current_exp)
                    self.agent.update_w(current_exp)
                    self.agent.update_r(current_exp)
                    self.agent.update_eligibility()
                else:
                    raise ValueError("agent_type should be Q, SF or PF")
                current_state = next_state
                self.cumulative_reward = self.cumulative_reward + reward
                if done:
                    break
                # step_idx += 1

            self.agent.decay_epsilon()

            self.step_lengths.append(step_idx + 1)
            self.rewards.append(self.cumulative_reward)
            self.Q_error_history.append(self.compute_q_error())
        return self.step_lengths, self.rewards, self.Q_error_history

    def reset(self):
        self.agent.reset()
        self.cumulative_reward = 0
        self.step_lengths = []
        self.rewards = []
        self.Q_error_history = []


class training_loop_2D:
    def __init__(
        self, env, agent, agent_type: str = "Q", episodes=500, max_step_length=100
    ):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_step_length = max_step_length

        self.step_lengths = []
        self.rewards = []
        self.Q_history = []
        self.agent_type = agent_type
        self.cumulative_reward = 0

        self.agent.reset()
        # compute true q values works only for my custom 1D env
        # should be avoided for other envs like neuro-nav
        # self.true_q_values = compute_true_q_value(self.env, gamma=self.agent.gamma)

    def run(self):
        for i in range(self.episodes):
            current_state = self.env.reset()
            done = False
            # reward_error = []

            # step_idx = 0 for while loop
            # while True:
            for step_idx in range(self.max_step_length):
                action = self.agent.sample_action(current_state)
                next_state, reward, done, _ = self.env.step(action)
                current_exp = (current_state, action, reward, next_state, done)
                if self.agent_type == "Q" or self.agent_type == "Q(lambda)":
                    self.agent.update_q_weights(current_exp)
                elif self.agent_type == "SF":
                    self.agent.update_w(current_exp)
                    self.agent.update_r(current_exp)
                elif self.agent_type == "PF":
                    self.agent.input(current_exp)
                    self.agent.update_w(current_exp)
                    self.agent.update_r(current_exp)
                    self.agent.update_eligibility()
                else:
                    raise ValueError("agent_type should be Q, SF or PF")
                current_state = next_state
                self.cumulative_reward = self.cumulative_reward + reward
                if done:
                    break
                # step_idx += 1

            self.agent.decay_epsilon()

            self.step_lengths.append(step_idx + 1)
            self.rewards.append(self.cumulative_reward)
            self.Q_history.append(copy.deepcopy(self.agent.q_table))
        return self.step_lengths, self.rewards, self.Q_history

    def reset(self):
        self.agent.reset()
        self.cumulative_reward = 0
        self.step_lengths = []
        self.rewards = []
        self.Q_history = []
        # self.Q_error_history = []
