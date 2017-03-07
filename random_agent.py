import cv2
import numpy as np

from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0

        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        idx = np.random.choice(4)
        self.total_reward += self.move(self.idx2act[idx])

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        cv2.imshow("Enduro", self._image)
        cv2.waitKey(40)


if __name__ == "__main__":
    a = RandomAgent()
    a.run(False, episodes=2, draw=True)
    print 'Total reward: ' + str(a.total_reward)
