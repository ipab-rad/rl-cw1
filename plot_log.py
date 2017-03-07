import numpy as np
import pickle

import matplotlib.pylab as plt

log = pickle.load(open("log.p", "rb"))

plt.subplot(211)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
curve, = plt.plot([e[1][-1] for e in log], 'b')

plt.subplot(212)
plt.title("Greedy Policy")
plt.xlabel("Agent position")
plt.ylabel("Opponent position")
base_cmap = plt.cm.get_cmap('GnBu')
cm = base_cmap.from_list('GnBu4', base_cmap(np.linspace(0, 1, 4)), 4)
plt.gca().set_xticks(np.arange(-.5, 10, 1.0), minor=True)
plt.gca().set_yticks(np.arange(-.5, 11, 1.005), minor=True)
plt.grid(which='minor')
im = plt.imshow(np.argmax(log[-1][2], axis=2).transpose(),
                interpolation='nearest', cmap=cm, origin='lower')

cbar = plt.colorbar()
cbar.set_clim(0, 3)
cbar.set_ticks([0, 0.375, 1.125, 1.875, 2.625, 3])
cbar.set_ticklabels(['', 'Accelerate', 'Right', 'Left', 'Brake', ''])

total_rewards = [e[1][-1] for e in log]
empty_rewards = [None] * len(log)
for i in xrange(len(log)):
    curve.set_ydata(total_rewards[:i] + empty_rewards[i:])
    im.set_data(np.argmax(log[i][2], axis=2).transpose())
    plt.draw()
    plt.pause(0.001)

plt.show()
