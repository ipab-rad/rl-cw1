#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import copy
import cv2
import numpy as np
from ale_python_interface import ALEInterface

import enduro.state as es
import enduro.control as ctrl

def getScreenImage(ale, scale=None):
    [w, h] = ale.getScreenDims()

    if scale is not None:
        w = int(w * scale)
        h = int(h * scale)

    image = cv2.cvtColor(ale.getScreenRGB(), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)

    return image

def getSymbolicRepresentation(ale):
    img = getScreenImage(ale)
    extractor = es.StateExtractor()
    state = extractor.run(img, draw=True, scale=4.0)

    cv2.imshow("Image", img)
    cv2.imshow("State", state.draw())

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)
# ale.setBool('sound', True)
# ale.setBool('display_screen', True)
# ale.setBool('color_averaging', True)
ale.setFloat('repeat_action_probability', 0.0)

# Load the ROM file
ale.loadROM('roms/enduro.bin')
# ale.loadROM('roms/skiing.bin')

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
print([ctrl.Action.toString(a) for a in legal_actions])
controller = ctrl.Controller(ale)

# Play 10 episodes
for episode in range(10):
    total_reward = 0

    while not ale.game_over():
        # a = legal_actions[randrange(len(legal_actions))]
        # Apply an action and get the resulting reward
        # reward = ale.act(ctrl.Action.LEFT_FIRE)
        # print reward
        # total_reward += reward

        key = cv2.waitKey(0)
        if chr(key & 255) == 'a':
            controller.move(ctrl.Action.LEFT)
        if chr(key & 255) == 'd':
            controller.move(ctrl.Action.RIGHT)
        if chr(key & 255) == 'w':
            controller.move(ctrl.Action.FIRE)
        if chr(key & 255) == 's':
            controller.move(ctrl.Action.DOWN)
        if chr(key & 255) == 'q':
            break

        getSymbolicRepresentation(ale)


    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
