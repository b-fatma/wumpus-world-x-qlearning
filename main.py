from WumpusWorld import WumpusWorld
from QLearning import QLearningAgent

if __name__ == "__main__":
    env = WumpusWorld()
    reward_table = env.create_reward_table()

    agent = QLearningAgent(env)
    Q = agent.train()

    policy = agent.get_policy()
    print(policy)