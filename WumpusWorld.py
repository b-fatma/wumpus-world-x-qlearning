import numpy as np

class WumpusWorld:
    def __init__(self, size=4):
        self.size = size
        self.num_states = size * size
        self.reset()

    def reset(self):
        self.initial_pos = (0, 0)
        self.gold_pos = (2, 1)
        self.pits = {(0, 2), (2, 2), (3, 3)}
        self.wumpus_pos = (2, 0)
        self.exit_pos = [self.gold_pos, self.wumpus_pos, *self.pits]
        return self.get_state(self.initial_pos)

    def get_state(self, pos):
        x, y = pos
        return x * self.size + y

    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_adjacent_states(self, pos):
        x, y = pos
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        adjacent = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                adjacent.append(self.get_state((nx, ny)))
        return adjacent

    def create_reward_table(self):
        R = np.zeros((self.num_states, self.num_states))  # default reward for non valid states

        for x in range(self.size):
            for y in range(self.size):
                pos = (x, y)
                s_prime = self.get_state(pos)
                adjacent_states = self.get_adjacent_states(pos)

                for s in adjacent_states:
                    if pos == self.gold_pos:
                        R[s, s_prime] = 1000
                    elif pos == self.wumpus_pos:
                        R[s, s_prime] = -10000
                    elif pos in self.pits:
                        R[s, s_prime] = -10
                    else:
                        R[s, s_prime] = 10  # safe
        return R

                    
