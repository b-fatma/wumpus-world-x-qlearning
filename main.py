from WumpusWorld import WumpusWorld
from QLearning import QLearningAgent

if __name__ == "__main__":
    env = WumpusWorld()
    print(env.create_reward_table())
    agent = QLearningAgent(env)
    Q = agent.train()
    print(Q)
    policy = agent.get_policy()
    print(policy)