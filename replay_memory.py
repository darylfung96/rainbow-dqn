import random

from segment_tree import SegmentTree


class ReplayMemory:

    def __init__(self, max_memory=1000):
        self.max_memory = max_memory
        self.memory = SegmentTree(max_memory)
        self._count = 0

    @property
    def count(self):
        return self._count

    def add_memory(self, state_input, best_action, reward, done, next_state_input, td):
        data = [state_input, best_action, reward, done, next_state_input]

        self.memory.add(td, data)

        if self._count <= self.max_memory:
            self._count += 1

    def get_memory(self, batch_size):
        segment = self.memory.max_td / batch_size

        batch_tree_index = []
        tds = []
        batch = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            segment = random.uniform(a, b)
            tree_index, td, data = self.memory.get(segment)
            batch_tree_index.append(tree_index)
            tds.append(td)
            batch.append(data)

        return batch_tree_index, tds, batch

    def update_memory(self, tree_indexes, tds):
        for i in range(len(tree_indexes)):
            self.memory.update(tree_indexes[i], tds[i])
