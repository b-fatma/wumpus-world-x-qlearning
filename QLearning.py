import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.R = env.create_reward_table()
        self.Q = np.zeros_like(self.R)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def get_valid_next_states(self, state):
        # Valid transitions are those with non null rewards 
        return [s_prime for s_prime in range(self.env.num_states) if self.R[state, s_prime] > 0]

    def choose_action(self, state):
        valid_states = self.get_valid_next_states(state)

        if not valid_states:
            return None  # No move possible

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(valid_states)
        else:
            # Choose the best next state
            q_values = [self.Q[state, s_prime] for s_prime in valid_states]
            max_q = max(q_values)
            best_next_states = [s_prime for s_prime in valid_states if self.Q[state, s_prime] == max_q]
            return random.choice(best_next_states)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            steps = 0

            while True:
                action = self.choose_action(state)
                if action is None:
                    break  # no valid moves

                reward = self.R[state, action]
                next_valid = self.get_valid_next_states(action)
                max_q_next = max([self.Q[action, s_next] for s_next in next_valid], default=0)

                # Q-learning update
                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward + self.gamma * max_q_next - self.Q[state, action]
                )

                state = action
                steps += 1

                if reward == 1000 or reward == -10000 or reward == -10:
                    break  # terminal state (gold, wumpus, or pit)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1} completed in {steps} steps.")

        return self.Q

    def get_policy(self):
        policy = {}
        for state in range(self.env.num_states):
            valid_states = self.get_valid_next_states(state)
            if valid_states:
                best = max(valid_states, key=lambda s_prime: self.Q[state, s_prime])
                policy[state] = best
        return policy
