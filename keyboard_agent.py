import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class KeyboardAgent(Agent):
    def __init__(self):
        super(KeyboardAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0

        cv2.imshow("Enduro", self._image)
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        key = cv2.waitKey(0)
        action = Action.NOOP
        if chr(key & 255) == 'a':
            action = Action.LEFT
        if chr(key & 255) == 'd':
            action = Action.RIGHT
        if chr(key & 255) == 'w':
            action = Action.ACCELERATE
        if chr(key & 255) == 's':
            action = Action.BREAK

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT or Action.ACCELERATE
        # Do not use plain integers between 0 - 3 as it will not work
        self.total_reward += self.move(action)

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
        # Show the latest game frame
        cv2.imshow("Enduro", self._image)

if __name__ == "__main__":
    a = KeyboardAgent()
    a.run(False, episodes=2, draw=True)
    print 'Total reward: ' + str(a.total_reward)
