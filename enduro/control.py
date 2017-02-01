from enduro.action import Action


class Controller:
    def __init__(self, ale):
        self._ale = ale

    def move(self, action):
        reward = 0
        repeat = 4 if action == Action.ACCELERATE else 8

        for i in range(repeat):
            reward += self._ale.act(action)

        return reward
