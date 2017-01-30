#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import cv2
import sys
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


ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)
# ale.setBool('sound', True)
ale.setBool('display_screen', True)
# ale.setBool('color_averaging', True)
ale.setFloat('repeat_action_probability', 0.0)

# Load the ROM file
rom_file = str.encode(sys.argv[1])
ale.loadROM('roms/enduro.bin')
# ale.loadROM('roms/skiing.bin')


# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
print([Action.toString(a) for a in legal_actions])

# Play 10 episodes
for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        # a = legal_actions[randrange(len(legal_actions))]
        # Apply an action and get the resulting reward
        reward = ale.act(Action.FIRE)
        total_reward += reward
        cv2.imshow("Image", cv2.cvtColor(ale.getScreenRGB(), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
