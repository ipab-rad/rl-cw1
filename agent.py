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


def removeOffroadRegions(image):
    [h, w, _] = image.shape

    offroad_pt = (int(0.99 * image.shape[1]), int(0.5 * image.shape[0]))
    offroad_color = image[offroad_pt[1], offroad_pt[0], :]
    mask = np.logical_or(
        np.logical_or(image[:, :, 0] != offroad_color[0],
                      image[:, :, 1] != offroad_color[1]),
        image[:, :, 2] != offroad_color[2])
    return image * mask.reshape(image.shape[0], image.shape[1], -1)


def intersectRoad(image, hline):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzeros = np.nonzero(img[hline, :])[0]

    # Make sure to hande aliasing effects
    if len(nonzeros) < 2:
        return intersectRoad(image, hline - 1)

    return (nonzeros[0], nonzeros[-1])


def detectRoadGrid(image):
    grid = []
    for rel_h in [0.3, 0.33, 0.37, 0.42, 0.48, 0.55, 0.63, 0.725]:
        line = []
        y = int(rel_h * image.shape[0])
        [l, r] = intersectRoad(image, y)
        for rel_w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            x = int((1 - rel_w) * l + rel_w * r)
            line.append([x, y])
        grid.append(line)
    return grid


def getRoadMask(image, grid):
    g = copy.deepcopy(grid)
    # Shring grid to avoid noise on the road edges
    for i in range(len(g)):
        g[i][0][0] += max(5, i)
        g[i][-1][0] -= max(5, i)

    # Make a polygon from the grid
    pts = []
    # Top
    for i in range(len(g[0])):
        pts.append(g[0][i])
    # Right
    for i in range(len(g)):
        pts.append(g[i][-1])
    # Bottom
    for i in reversed(range(len(g[-1]))):
        pts.append(g[-1][i])
    # Left
    for i in reversed(range(len(g))):
        pts.append(g[i][0])

    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.fillPoly(mask, np.asarray([pts]), (1))
    mask = mask.reshape(mask.shape[0], mask.shape[1], -1)

    return mask


def detectCars(image):
    res = {}

    # Find player's car pixels
    mask = np.logical_and(
        np.logical_and(image[:, :, 0] > 180,
                       image[:, :, 1] > 180),
        image[:, :, 2] > 180).reshape(image.shape[0], image.shape[1], -1)

    # Detect the contour of the player's car
    _, thresh = cv2.threshold(
        cv2.cvtColor(image * mask, cv2.COLOR_BGR2GRAY), 170, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Make sure player's car is found
    if len(contours) == 0:
        return res
    else:
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    res["self"] = cv2.boundingRect(contours[0])

    # Detect the other cars
    mask = np.logical_not(mask)
    _, thresh = cv2.threshold(
        cv2.cvtColor(image * mask, cv2.COLOR_BGR2GRAY), 64, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res["others"] = [cv2.boundingRect(c) for c in contours]
    return res


def getState(grid, cars):
    def center(rect):
        return (rect[0] + int(0.5 * rect[2]),
                rect[1] + int(0.5 * rect[3]))

    def inCell(i, j, pt):
        c = np.asarray(
            [grid[i][j], grid[i + 1][j],
             grid[i + 1][j + 1], grid[i][j + 1]])
        return cv2.pointPolygonTest(c, pt, False) >= 0

    def getCell(pt):
        for i in range(len(grid) - 1):
            for j in range(len(grid[i]) - 1):
                if inCell(i, j, pt):
                    return (i, j)
        return None

    state = {}
    state["grid"] = np.zeros((len(grid) - 1, len(grid[0]) - 1), np.uint8)
    (pos_y, pos_x) = getCell(center(cars["self"]))
    pos_y = len(grid) - pos_y - 2
    state["position"] = (pos_y, pos_x)
    state["grid"][pos_y, pos_x] = 2

    for c in cars["others"]:
        (pos_y, pos_x) = getCell(center(c))
        pos_y = len(grid) - pos_y - 2
        state["grid"][pos_y, pos_x] = 1
    return state


def drawRoadGrid(image, grid, scale=1.0):
    for i in range(len(grid) - 1):
        cv2.line(image,
                 tuple(np.asarray(grid[i][0]) * scale),
                 tuple(np.asarray(grid[i][-1]) * scale),
                 (0xE8, 0xE8, 0xE8))
        for j in range(len(grid[i])):
            cv2.line(image,
                     tuple(np.asarray(grid[i][j]) * scale),
                     tuple(np.asarray(grid[i+1][j]) * scale),
                     (0xE8, 0xE8, 0xE8))
    cv2.line(image,
             tuple(np.asarray(grid[-1][0]) * scale),
             tuple(np.asarray(grid[-1][-1]) * scale),
             (0xE8, 0xE8, 0xE8))
    return image


def drawCars(image, cars, scale=1.0, margin=0):
    def tl(r, d=margin):
        return (scale * r[0] - d, scale * r[1] - d)

    def br(r, d=margin):
        return (scale * (r[0] + r[2]) + d, scale * (r[1] + r[3]) + d)

    r = cars["self"]
    cv2.rectangle(image, tl(r), br(r), (0x0F, 0x70, 0x50), 2)

    for r in cars["others"]:
        cv2.rectangle(image, tl(r), br(r), (0x43, 0x04, 0xAE), 2)
    return image


def drawState(state):
    grid = state["grid"]
    [h, w] = grid.shape
    sz = 50
    image = np.ones((h * sz, w * sz, 3), np.uint8)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 2:
                cv2.rectangle(image,
                              (j * sz, i * sz),
                              ((j + 1) * sz, (i + 1) * sz),
                              (0x0F, 0x70, 0x50),
                              -1)
            elif grid[i][j] == 1:
                cv2.rectangle(image,
                              (j * sz, i * sz),
                              ((j + 1) * sz, (i + 1) * sz),
                              (0x43, 0x04, 0xAE),
                              -1)
    for i in range(len(grid)):
        cv2.line(image,
                 (0, i * sz),
                 (image.shape[1], i * sz),
                 (0xE8, 0xE8, 0xE8), 1)

    for i in range(len(grid)):
        cv2.line(image,
                 (i * sz, 0),
                 (i * sz, image.shape[0]),
                 (0xE8, 0xE8, 0xE8), 1)
    return cv2.flip(image, 0)


def getSymbolicRepresentation(ale):
    img = getScreenImage(ale)
    img = removeOffroadRegions(img)
    road_grid = detectRoadGrid(img)
    road_mask = getRoadMask(img, road_grid)
    cars = detectCars(img * road_mask)

    if "self" not in cars:
        return

    state = getState(road_grid, cars)

    # Visualistaion
    img = cv2.resize(img, (4 * img.shape[1], 4 * img.shape[0]))
    img = drawRoadGrid(img, road_grid, 4)
    img = drawCars(img, cars, 4, 5)
    cv2.imshow("Image", img)
    cv2.imshow("State", drawState(state))

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
