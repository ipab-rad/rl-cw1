import copy
import cv2
import numpy as np


class EnvironmentState:
    @staticmethod
    def draw(grid, sz=40):
        [h, w] = grid.shape
        image = np.ones((h * sz, w * sz, 3), np.uint8)
        for i in range(h):
            for j in range(w):
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
        for i in range(h):
            cv2.line(image,
                     (0, i * sz),
                     (image.shape[1], i * sz),
                     (0xE8, 0xE8, 0xE8), 1)

        for i in range(w):
            cv2.line(image,
                     (i * sz, 0),
                     (i * sz, image.shape[0]),
                     (0xE8, 0xE8, 0xE8), 1)

        return cv2.flip(image, 0)


class StateExtractor:
    def __init__(self, ale):
        self._ale = ale
        self._is_cv3 = cv2.__version__.startswith("3.")

    def run(self, draw=False, scale=1.0):
        image = self.__getScreenImage()
        # Use an image copy for processing
        img = self.__removeOffroadRegions(np.copy(image))
        self._road_grid = self.__detectRoadGrid(img)
        self._cars = self.__detectCars(
            img * self.__getRoadMask(img, self._road_grid))

        self._state_grid = self.__getStateGrid(self._road_grid, self._cars)

        if draw:
            self.__draw(image, scale)

        return (self._state_grid, image)

    def __getScreenImage(self):
        [w, h] = self._ale.getScreenDims()
        image = cv2.cvtColor(self._ale.getScreenRGB(), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_NEAREST)
        return image

    def __removeOffroadRegions(self, image):
        # This pixel always has the color of the background
        offroad_pt = (int(0.99 * image.shape[1]), int(0.5 * image.shape[0]))
        offroad_color = image[offroad_pt[1], offroad_pt[0], :]
        mask = np.logical_or(
            np.logical_or(image[:, :, 0] != offroad_color[0],
                          image[:, :, 1] != offroad_color[1]),
            image[:, :, 2] != offroad_color[2])
        return image * mask.reshape(image.shape[0], image.shape[1], -1)

    def __intersectRoad(self, image, hline):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nonzeros = np.nonzero(img[hline, :])[0]

        # Make sure to hande aliasing effects
        if len(nonzeros) < 2:
            return self.__intersectRoad(image, hline + 1)

        return (nonzeros[0], nonzeros[-1])

    def __detectRoadGrid(self, image):
        grid = []

        hs = [0.33, 0.34, 0.36, 0.38, 0.4, 0.43,
              0.46, 0.49, 0.53, 0.57, 0.63, 0.7]

        for rel_h in hs:
            line = []
            y = int(rel_h * image.shape[0])
            [l, r] = self.__intersectRoad(image, y)
            for rel_w in [0.01 * x for x in range(0, 101, 10)]:
                x = int((1 - rel_w) * l + rel_w * r)
                line.append([x, y])
            grid.append(line)
        return grid

    def __getRoadMask(self, image, grid):
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

    def __detectCars(self, image):
        res = {}

        # Find player's car pixels
        mask = np.logical_and(
            np.logical_and(image[:, :, 0] > 180,
                           image[:, :, 1] > 180),
            image[:, :, 2] > 180).reshape(image.shape[0], image.shape[1], -1)

        # Detect the contour of the player's car
        _, thresh = cv2.threshold(
            cv2.cvtColor(image * mask, cv2.COLOR_BGR2GRAY), 170, 255, 0)

        if self._is_cv3:
            _, contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Make s
        if len(contours) == 0:
            return res
        else:
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        res["self"] = cv2.boundingRect(contours[0])

        # Detect the other cars
        mask = np.logical_not(mask)
        _, thresh = cv2.threshold(
            cv2.cvtColor(image * mask, cv2.COLOR_BGR2GRAY), 64, 255, 0)

        if self._is_cv3:
            _, contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        res["others"] = [cv2.boundingRect(c) for c in contours]
        return res

    def __getStateGrid(self, grid, cars):
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

        state = np.zeros((len(grid) - 1, len(grid[0]) - 1), np.uint8)
        (pos_y, pos_x) = getCell(center(cars["self"]))
        pos_y = len(grid) - pos_y - 2
        state[pos_y, pos_x] = 2

        for c in cars["others"]:
            (pos_y, pos_x) = getCell(center(c))
            pos_y = len(grid) - pos_y - 2

            # Colision ocurred
            while state[pos_y, pos_x] == 2:
                pos_y += 1

            state[pos_y, pos_x] = 1

        return state

    def __drawRoadGrid(self, image, grid, scale=1.0):
        for i in range(len(grid) - 1):
            cv2.line(image,
                     tuple((np.asarray(grid[i][0]) * scale).astype(int)),
                     tuple((np.asarray(grid[i][-1]) * scale).astype(int)),
                     (0xE8, 0xE8, 0xE8))
            for j in range(len(grid[i])):
                cv2.line(image,
                         tuple((np.asarray(grid[i][j]) * scale).astype(int)),
                         tuple((np.asarray(grid[i+1][j]) * scale).astype(int)),
                         (0xE8, 0xE8, 0xE8))
        cv2.line(image,
                 tuple((np.asarray(grid[-1][0]) * scale).astype(int)),
                 tuple((np.asarray(grid[-1][-1]) * scale).astype(int)),
                 (0xE8, 0xE8, 0xE8))

    def __drawCars(self, image, cars, scale=1.0, margin=0):
        def tl(r, d=margin):
            return (int(scale * r[0]) - d, int(scale * r[1]) - d)

        def br(r, d=margin):
            return (int(scale * (r[0] + r[2])) + d,
                    int(scale * (r[1] + r[3])) + d)

        r = cars["self"]
        cv2.rectangle(image, tl(r), br(r), (0x0F, 0x70, 0x50), 2)

        for r in cars["others"]:
            cv2.rectangle(image, tl(r), br(r), (0x43, 0x04, 0xAE), 2)

    def __draw(self, image, scale):
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
        if scaled_image.shape != image.shape:
            image.resize(scaled_image.shape, refcheck=False)
        np.copyto(image, scaled_image)
        self.__drawRoadGrid(image, self._road_grid, scale)
        self.__drawCars(image, self._cars, scale, 5)
