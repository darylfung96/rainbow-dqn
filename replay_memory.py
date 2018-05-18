

class ReplayMemory:

    def __init__(self, max_memory=1000):
        self.max_memory = max_memory
        self.memory = []

    def add_memory(self, state_input, reward, done, next_state_input):
        if len(self.memory) > self.max_memory:
            del self.memory[0]

        self.memory.append([state_input, reward, done, next_state_input])

    def get_memory(self, memory_size):
        pass