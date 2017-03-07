import cv2
import numpy as np
import pickle

from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # The horizon defines how far the agent can see
        self.horizon_row = 5

        self.grid_cols = 10
        # The state is defined as a tuple of the agent's x position and the
        # x position of the closest opponent which is lower than the horizon,
        # if any is present. There are four actions and so the Q(s, a) table
        # has size of 10 * (10 + 1) * 4 = 440.
        self.Q = np.ones((self.grid_cols, self.grid_cols + 1, 4))

        # Add initial bias toward moving forward. This is not necessary,
        # however it speeds up learning significantly, since the game does
        # not provide negative reward if no cars have been passed by.
        self.Q[:, :, 0] += 1.

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}

        # Learning rate
        self.alpha = 0.01
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01

        # Log the obtained reward during learning
        self.last_episode = 1
        self.episode_log = np.zeros(6510) - 1.
        self.log = []

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        self.total_reward = 0
        self.next_state = self.buildState(grid)

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        self.state = self.next_state

        # If exploring
        if np.random.uniform(0., 1.) < self.epsilon:
            # Select a random action using softmax
            Q_s = self.Q[self.state[0], self.state[1], :]
            probs = np.exp(Q_s) / np.sum(np.exp(Q_s))
            idx = np.random.choice(4, p=probs)
            self.action = self.idx2act[idx]
        else:
            # Select the greedy action
            self.action = self.idx2act[self.argmaxQsa(self.state)]

        self.reward = self.move(self.action)
        self.total_reward += self.reward

    def sense(self, grid):
        self.next_state = self.buildState(grid)

        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        # Read the current state-action value
        Q_sa = self.Q[self.state[0], self.state[1], self.act2idx[self.action]]

        # Calculate the updated state action value
        Q_sa_new = Q_sa + self.alpha * (self.reward + self.gamma * self.maxQsa(self.next_state) - Q_sa)

        # Write the updated value
        self.Q[self.state[0], self.state[1], self.act2idx[self.action]] = Q_sa_new

    def callback(self, learn, episode, iteration):
        if not iteration % 1000:
            print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)

        # Initialise the log for the next episode
        if episode != self.last_episode:
            iters = np.nonzero(self.episode_log >= 0)
            rewards = self.episode_log[iters]
            self.log.append((np.asarray(iters).flatten(), rewards, np.copy(self.Q)))
            self.last_episode = episode
            self.episode_log = np.zeros(6510) - 1.

        # Log the reward at the current iteration
        self.episode_log[iteration] = self.total_reward

        if not episode % 100:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(20)

    def buildState(self, grid):
        state = [0, 0]

        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        state[0] = x

        # Sum the rows of the grid
        rows = np.sum(grid, axis=1)
        # Ignore the agent
        rows[0] -= 2
        # Get the closest row where an opponent is present
        rows = np.sort(np.argwhere(rows > 0).flatten())

        # If any opponent is present
        if rows.size > 0:
            # Add the x position of the first opponent on the closest row
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:
                    # 0 means that no agent is present and so
                    # the index is offset by 1
                    state[1] = i + 1
                    break
        return state

    def maxQsa(self, state):
        return np.max(self.Q[state[0], state[1], :])

    def argmaxQsa(self, state):
        return np.argmax(self.Q[state[0], state[1], :])


if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=500, draw=True)
    pickle.dump(a.log, open("log.p", "wb"))
