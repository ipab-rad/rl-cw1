import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # Initialise your member variables here

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """
        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # self.move(action) executes the action and returns the reward signal
        pass

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("State", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs any learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        # Show the latest game frame
        cv2.imshow("Enduro", self._image)


if __name__ == "__main__":
    a = QAgent()
    a.run(episodes=1, draw=True)
    print 'Total reward: ' + str(a.total_reward)
