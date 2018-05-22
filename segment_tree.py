import numpy as np


class SegmentTree:

    def __init__(self, capacity):

        # the tree is build accordingly:
        #       parents       leaves
        # { capacity + 1 } { capacity }
        # total size will be capacity + 1 + capacity
        # tree will store the td error to prioritize the replay
        self.tree = np.zeros(2 * capacity + 1)
        # data will store [state, reward, next_state, done]
        self.data = np.empty(capacity, dtype=np.object)
        self.capacity = capacity
        self.data_index = 0
        self.max_td = 1 # set the max p as 1 first

    def add(self, td, data):
        """

        :param td:   temporal difference error
        :param data: [state, reward, next_state, done]
        :return: None
        """
        tree_index = self.data_index + self.capacity + 1
        self.data[self.data_index] = data

        self.update(tree_index, td)
        self.data_index += 1

        if self.data_index > self.capacity:
            self.data_index = 0

        self.max_td = max(td, self.max_td)

    def update(self, tree_index, td):
        change = td - self.tree[tree_index]
        self.tree[tree_index] = td

        while True:
            tree_index = (tree_index-1) // 2
            self.tree[tree_index] += change
            if tree_index == 0:
                break

    def get(self, value):
        tree_index = self.retrieve(value)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.data[data_index]

    def retrieve(self, value):
        starting_tree_index = 0

        while True:
            l_index = starting_tree_index * 2 + 1
            r_index = l_index + 1

            # return the leaf node
            if l_index >= len(self.tree):
                return starting_tree_index

            if value < self.tree[l_index]:
                starting_tree_index = l_index
            else:
                starting_tree_index = r_index
