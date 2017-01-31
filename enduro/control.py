from ale_python_interface import ALEInterface

class Action:
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UP_RIGHT = 6
    UP_LEFT = 7
    DOWN_RIGHT = 8
    DOWN_LEFT = 9
    UP_FIRE = 10
    RIGHT_FIRE = 11
    LEFT_FIRE = 12
    DOWN_FIRE = 13
    UP_RIGHT_FIRE = 14
    UP_LEFT_FIRE = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE = 17

    @staticmethod
    def toString(a):
        return ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
                "UP-RIGHT", "UP-LEFT", "DOWN-RIGHT", "DOWN-LEFT",
                "UP-FIRE", "RIGHT-FIRE", "LEFT-FIRE", "DOWN-FIRE",
                "UP-RIGHT-FIRE", "UP-LEFT-FIRE",
                "DOWN-RIGHT-FIRE", "DOWN-LEFT-FIRE"][a]

class Controller:
    def __init__(self, ale):
        self._ale = ale

    def move(self, action):
        reward = 0
        for i in range(10):
            reward += self._ale.act(action)
        return reward
