# Reinforcement Learning: Coursework 1

## Writing your own agent

#### The `Agent` class

In order to implement your own Enduro agent you should derive from the `Agent` class which provides the folloing functions:

* `def run(self, learn, episodes)`  
Implements the playing/learning loop and calls the corresponding functions implemented by the sublclasses. If `learn` is set to `True` then the `learn()` will be called at every time step. `episodes` is the number of episodes for which the agent should be run.

* `def getActionsSet(self)`  
Returns the set of possible actions: `[Action.ACCELERATE, Action.RIGHT, Action.LEFT, Action.BREAK]`

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
The environment grid is a 11x10 Numpy array, where a cell contains 2 if the agent is at that position, 1 if there is an opponent car at that postion or 0 if the space is free. Your agent is always at row 0 while the most distant opponents are at row 10. The leftmost position on the road corresponds to column 0 while the rightmost one - to column 9.

## Example
A simple keyboard controlled agent is provided as an example. You can run it with
```
python keyboard_agent.py
```
Make sure you run the command from the root directory of this repository.

## Solution & Results

The state representation for the Q-learning agent of the proposed solution is a 2-dimensional vector which contains the x-cooridinate of the agent and the x-cooridinate of the closest opponent car incremented by one. If no opponents are present then the second component of the state vector is set to 0. During action selection, the agent is epsilon greedy with `epsilon=0.01` where a random action is selected through softmax sampling. After learning for 500 episodes the following results are obtained:

![Results](https://raw.githubusercontent.com/ipab-rad/rl-cw1/master/figs/results.svg)

You can inspect the results with:

```
python plot_log.py
```

## Setup & Requirements
This pacakge should run out of the box on a DICE machine, however if you want to install it on your own computer then you should have OpenCV 2 and the Arcade Learning Environment installed. OpenCV installation is OS and distribution dependent, so you should find out how to do it for your own system. Installation instructions for the Arcade Learning Environment can be found [here](https://github.com/mgbellemare/Arcade-Learning-Environment#quick-start). Make sure you use Python2 for running your agents.

### Setup on Ubuntu 16.04
These are the steps you should follow in order to setup OpenCV, ALE and the coursework package on a clean Ubuntu 16.04. You might want to use them to prepare a virtual machine and work on it, instead of a DICE machine.

```
sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3.5-dev python-pip
sudo pip install numpy

cd ~
wget -O opencv-3.2.0.zip https://github.com/Itseez/opencv/archive/3.2.0.zip
unzip opencv-3.2.0.zip 
cd opencv-3.2.0/
mkdir build
cd build/
cmake ..
make -j4
sudo make install

cd ~
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
mkdir build
cd build/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j4
cd ..
sudo pip install .

cd ~
git clone https://github.com/ipab-rad/rl-cw1.git
cd rl-cw1/
python keyboard_agent.py 
```

