# agents.py

import numpy as np
import predecessor.utils as utils


class QLearningAgent:
    def __init__(
        self,
        featvec_size: int,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
    ):
        self.featvec_size = featvec_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.decay_rate = 0.99
        self.q_weights = np.zeros((self.action_size, self.featvec_size))

    def estimated_q(self, featvec):
        return np.matmul(self.q_weights, featvec)

    def sample_action(self, featvec):
        q_values = self.estimated_q(featvec)
        # e-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = utils.my_argmax(q_values)
        return action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    def update_q_weights(self, current_exp):
        current_state = current_exp[0]
        action = current_exp[1]
        reward = current_exp[2]
        next_state = current_exp[3]
        done = current_exp[4]
        if done:
            td_error = reward - self.estimated_q(current_state)[action]
        else:
            td_error = (
                reward
                + self.gamma * self.estimated_q(next_state).max()
                - self.estimated_q(current_state)[action]
            )
        delta_q_weights = self.alpha * td_error * current_state
        self.q_weights[action, :] += delta_q_weights
        return delta_q_weights

    def reset(self):
        self.q_weights = np.zeros((self.action_size, self.featvec_size))
        self.epsilon = self.init_epsilon
        return self.q_weights

    @property
    def q_table(self):
        return self.q_weights


class QLearningLambdaAgent(QLearningAgent):
    def __init__(
        self, featvec_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, lambda_=0.9
    ):
        super().__init__(featvec_size, action_size, alpha, gamma, epsilon)
        self.lambda_ = lambda_
        self.e_trace = np.zeros((self.action_size, self.featvec_size))

    def update_q_weights(self, current_exp):
        current_state = current_exp[0]
        action = current_exp[1]
        reward = current_exp[2]
        next_state = current_exp[3]
        done = current_exp[4]
        if done:
            td_error = reward - self.estimated_q(current_state)[action]
        else:
            td_error = (
                reward
                + self.gamma * self.estimated_q(next_state).max()
                - self.estimated_q(current_state)[action]
            )
        self.e_trace *= self.gamma * self.lambda_
        self.e_trace[action] += current_state
        self.q_weights += self.alpha * td_error * self.e_trace


class SFAgent:
    def __init__(
        self,
        featvec_size: int,
        action_size: int,
        alpha_r: float = 0.1,
        alpha_w: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
    ):
        self.featvec_size = featvec_size
        self.action_size = action_size
        self.sf_size = featvec_size

        self.alpha_r = alpha_r
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.decay_rate = 0.99
        self.w_matrix = np.stack(
            [np.eye(self.featvec_size) for i in range(self.action_size)]
        )
        self.r_vector = np.zeros(self.featvec_size)

    def estimated_sf_vec(self, featvec):
        est_sf_vec = self.w_matrix @ featvec
        return est_sf_vec

    def estimated_q(self, featvec):
        return np.matmul(self.estimated_sf_vec(featvec), self.r_vector)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    def sample_action(self, featvec):
        q_values = self.estimated_q(featvec)
        # e-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = utils.my_argmax(q_values)
        return action

    def update_w(self, current_exp):
        current_state = current_exp[0]
        action = current_exp[1]
        next_state = current_exp[3]
        sf_s_t = self.estimated_sf_vec(current_state)
        sf_s_t_1 = self.estimated_sf_vec(next_state)
        done = current_exp[4]
        if done:
            delta_in = next_state - sf_s_t[action, :]
        else:
            max_next_action = utils.my_argmax(self.estimated_q(next_state))
            delta_in = (
                current_state
                + self.gamma * sf_s_t_1[max_next_action, :]
                - sf_s_t[action, :]
            )
        delta_W = self.alpha_w * np.outer(delta_in, current_state)
        self.w_matrix[action, :, :] += delta_W
        return delta_W

    def update_r(self, current_exp):
        next_state = current_exp[3]
        reward = current_exp[2]
        delta_in = self.alpha_r * (reward - np.matmul(self.r_vector, next_state))
        delta_r_vector = delta_in * next_state
        self.r_vector += delta_r_vector
        return delta_r_vector

    @property
    def estimated_SR(self):
        feature_matrix = np.eye(self.featvec_size)
        return np.matmul(self.w_matrix, feature_matrix).T

    def get_policy(self):
        Q_matrix = self.w_matrix @ self.r_vector
        mask = Q_matrix == Q_matrix.max(0)
        greedy = mask / mask.sum(0)
        policy = (1 - self.epsilon) * greedy + (
            1 / self.action_size
        ) * self.epsilon * np.ones((self.action_size, self.featvec_size))

        return policy

    def get_SR_matrix(self):
        policy = self.get_policy()
        SR_matrix = np.diagonal(
            np.tensordot(policy.T, self.w_matrix, axes=1), axis1=1, axis2=2
        ).T
        return SR_matrix

    def reset(self):
        self.w_matrix = np.stack(
            [np.eye(self.featvec_size) for i in range(self.action_size)]
        )
        self.r_vector = np.zeros(self.featvec_size)
        self.epsilon = self.init_epsilon
        return self.w_matrix, self.r_vector

    @property
    def q_table(self):
        transposed_w_matrix = self.w_matrix.transpose(0, 2, 1)
        return np.matmul(transposed_w_matrix, self.r_vector)

    @property
    def sf_table(self):
        return self.w_matrix.transpose(0, 2, 1)


class PFAgent(SFAgent):
    def __init__(
        self,
        featvec_size: int,
        action_size: int,
        alpha_r: float = 0.1,
        alpha_w: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        lambda_: float = 0.8,
    ):
        super().__init__(featvec_size, action_size, alpha_r, alpha_w, gamma, epsilon)
        self.lambda_ = lambda_
        self.eligibility_state = np.zeros(self.featvec_size)
        self.w_matrix = np.stack(
            [np.eye(self.featvec_size) for i in range(self.action_size)]
        )
        self.r_vector = np.zeros(self.featvec_size)

    def input(self, current_exp):
        current_state = current_exp[0]
        self.eligibility_state += current_state
        return self.eligibility_state

    def update_w(self, current_exp):
        current_state = current_exp[0]
        action = current_exp[1]
        next_state = current_exp[3]
        pf_s_t = self.estimated_sf_vec(current_state)
        pf_s_t_1 = self.estimated_sf_vec(next_state)
        done = current_exp[4]
        if done:
            delta_in = next_state - pf_s_t[action, :]
        else:
            max_next_action = utils.my_argmax(self.estimated_q(next_state))

            delta_in = (
                current_state
                + self.gamma * pf_s_t_1[max_next_action, :]
                - pf_s_t[action, :]
            )
        delta_W = self.alpha_w * np.outer(delta_in, self.eligibility_state)
        self.w_matrix[action, :, :] += delta_W
        return delta_W

    def update_eligibility(self):
        self.eligibility_state = self.lambda_ * self.gamma * self.eligibility_state
        return self.eligibility_state

    def eligibility_reset(self):
        self.eligibility_state = np.zeros(self.featvec_size)
        return self.eligibility_state

    def reset(self):
        self.w_matrix = np.stack(
            [np.eye(self.featvec_size) for i in range(self.action_size)]
        )
        self.r_vector = np.zeros(self.featvec_size)
        self.epsilon = self.init_epsilon
        self.eligibility_state = np.zeros(self.featvec_size)
        return self.w_matrix, self.r_vector, self.eligibility_state
