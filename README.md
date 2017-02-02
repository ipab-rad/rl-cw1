# Reinforcement Learning: Coursework 1

## Writing your own agent

#### The `Agent` class

In order to implement your own Enduro agent you should derive from the `Agent` class which provides the folloing functions:

* `def run(self, learn, episodes)`  
Implements the playing/learning loop and calls the corresponding functions implemented by the sublclasses. If `learn` is set to `True` then the `learn()` will be called at every time step. `episodes` is the number of episodes for which the agent should be run.

* `def getActionsSet(self)`  
Returns the set of possible actions: `[Action.ACCELERATE, Action.LEFT, Action.RIGHT]`

* `def move(self, action)`  
Executes the `action`, advances the game to the next time step and **returns the received reward**.  
**NOTE:**  Make sure you use the action constants and **not** integers from 0 to 2, or any other encoding.

#### `Agent` sublclasses
The `Agent` class requires the following functions to be implemented by the subclass:

* `def initialise(self, grid)`  
This function is called at the beginning of each episode. It is useful for constructing the initial state from the [environment grid](#environment-grid).

* `def act(self)`  
This function is called at every iteration and it should implement the decision making process for selecting an action. You can execute an action and get the resulting reward with `reward = self.move(action)`.

* `def sense(self, grid)`  
This function is called at every iteration, after the `act()` function, and it should construct the new state from the updated [environment grid](#environment-grid).

* `def learn(self)`  
If the `learn` argument of `run()` functions is set to `True` this function is called at every iteration after the `act()` and `sense()` functions. It should implement the learning logic of the agent.

* `def callback(self, learn, episode, iteration)`  
This function is called at every iteration during the plying/learning loop and it is useful for debugging or reporting purposes. You have access to whether the agent is learning, the episode number as well as the iteration number.

In order to keep the `Agent` class as generic as possible it does not have member variables such as the current state, action, next state or reward. Your subclass should keep track of those and update them accordingly. Remember to initalise them in the constructor of your class.

## Environment grid
The environment grid is a 11x10 Numpy array, where a cell contains 1 if the agent is at that position, 2 if there is an opponent car at that postion or 0 if the space is free. Your agent is always at row 0 while the most distant opponents are at row 10. The leftmost position on the road corresponds to column 0 while the rightmost one - to column 9.

## Example
A simple keyboard controlled agent is provided as an example. You can run it with
```
python keyboard_agent.py
```
Make sure you run the command from the root directory of this repository.

## Questions

If you have any questions directly related to the code you can simply submit an issue to the repository.

## Setup & Requirements
This pacakge should run out of the box on a DICE machine, however if you want to install it on your own computer then you should have OpenCV 2 and the Arcade Learning Environment installed. OpenCV installation is OS and distribution dependent, so you should find out how to do it for your own system. Installation instructions for the Arcade Learning Environment can be found [here](https://github.com/mgbellemare/Arcade-Learning-Environment#quick-start). Make sure you use Python2 for running your agents.

