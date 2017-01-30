#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import cv2
import sys
import numpy as np
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

def getScreenImage(ale, scale=None):
    [w, h] = ale.getScreenDims()

    if scale is not None:
        w = int(w * scale)
        h = int(h * scale)

    image = cv2.cvtColor(ale.getScreenRGB(), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)

    return image

def intersectRoad(image, hline):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzeros = np.nonzero(img[hline, :])[0]

    # Make sure to hande aliasing effects
    if len(nonzeros) < 2:
        return intersectRoad(image, hline - 1)

    return (nonzeros[0], nonzeros[-1])


def getSymbolicRepresentation(ale):
    img = getScreenImage(ale, 4.0)
    [h, w, _] = img.shape
    horizon = int(0.3 * h)

    offroad_pt = (int(0.99 * w), int(0.5 * h))
    offroad_color = img[offroad_pt[1], offroad_pt[0], :]
    mask = np.logical_or(
        np.logical_or(img[:, :, 0] != offroad_color[0],
                      img[:, :, 1] != offroad_color[1]),
        img[:, :, 2] != offroad_color[2])
    img = img * mask.reshape(h, w, -1)


    # baselines = [int(0.65 * h), int(0.7 * h)]
    hlines = []
    vlines = []
    for rel_h in [0.3, 0.33, 0.37, 0.42, 0.48, 0.55, 0.63, 0.725]:
        hlines.append(int(rel_h * h))
        [l, r] = intersectRoad(img, hlines[-1])
        xs = []
        for rel_w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            xs.append(int(rel_w * l + (1 - rel_w) * r))
        vlines.append(xs)

    # Visualistaion
    for i in range(len(hlines) - 1):
        cv2.line(img,
                 (vlines[i][0], hlines[i]),
                 (vlines[i][-1], hlines[i]),
                 (255, 255, 255))
        for j in range(len(vlines[i])):
            cv2.line(img,
                     (vlines[i][j], hlines[i]),
                     (vlines[i + 1][j], hlines[i + 1]),
                     (255, 255, 255))

    cv2.line(img,
             (vlines[-1][0], hlines[-1]),
             (vlines[-1][-1], hlines[-1]),
             (255, 255, 255))
    cv2.imshow("Image", img);

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
print([Action.toString(a) for a in legal_actions])
# Play 10 episodes
for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        # a = legal_actions[randrange(len(legal_actions))]
        # Apply an action and get the resulting reward
        reward = ale.act(Action.FIRE)
        total_reward += reward

        # cv2.imshow("Image", getScreenImage(ale, 4.0));
        getSymbolicRepresentation(ale)
        key = cv2.waitKey(20)

    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
