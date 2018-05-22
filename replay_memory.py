import random

from segment_tree import SegmentTree

class ReplayMemory:

    def __init__(self, max_memory=1000):
        self.max_memory = max_memory
        self.memory = SegmentTree(max_memory)

    def add_memory(self, state_input, reward, done, next_state_input, td):
        data = [state_input, reward, done, next_state_input]

        self.memory.add(td, data)

    def get_memory(self, batch_size):
        segment = self.memory.max_td / batch_size

        batch_tree_index = []
        batch = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            segment = random.uniform(a, b)
            tree_index, data = self.memory.get(segment)
            batch_tree_index.append(tree_index)
            batch.append(data)

        return batch_tree_index, batch

    def update_memory(self):
        pass
