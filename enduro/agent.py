from ale_python_interface import ALEInterface
from enduro.action import Action
from enduro.control import Controller
from enduro.state import StateExtractor


class Agent(object):
    def __init__(self):
        self._ale = ALEInterface()
        self._ale.setInt('random_seed', 123)
        self._ale.setFloat('repeat_action_probability', 0.0)
        self._ale.loadROM('roms/enduro.bin')
        self._controller = Controller(self._ale)
        self._extractor = StateExtractor(self._ale)
        self._image = None

    def run(self, learn, episodes=1, draw=False):
        for e in range(episodes):
            # Observe the environment to set the initial state
            (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
            self.initialise(grid)

            num_frames = self._ale.getFrameNumber()

            # Each episode lasts 6500 frames
            while self._ale.getFrameNumber() - num_frames < 6500:
                # Take an action
                self.act()

                # Update the environment grid
                (grid, self._image) = self._extractor.run(draw=draw, scale=4.0)
                self.sense(grid)

                # Perform learning if required
                if learn:
                    self.learn()

                self.callback(learn, e + 1, self._ale.getFrameNumber() - num_frames)
            self._ale.reset_game()


    def getActionsSet(self):
        return [Action.ACCELERATE, Action.RIGHT, Action.LEFT]

    def move(self, action):
        return self._controller.move(action)

    def initialise(self, grid):
        raise NotImplementedErro

    def act(self):
        raise NotImplementedError

    def sense(self, grid):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def callback(self, learn, episode, iteration):
        raise NotImplementedError
