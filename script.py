import numpy as np
from matplotlib import pyplot as plt

f = open('score.txt', 'r')
lines = f.readlines()
l = []
for i in range(0, 80 * 2, 2):
    try:
        train = float(lines[i][-4:-1])
    except ValueError:
        train = float(lines[i][-2:-1])
        assert train == 1

    test = float(lines[i + 1][-9:-1])
    l.append([train, test])

score = np.array(l)
print(score)

plt.plot(np.arange(0, 4000, 50), score[:, 0], '+', label="Train accuracy")
plt.plot(np.arange(0, 4000, 50), score[:, 1], label="Validation accuracy")
plt.hlines(0.97435, 0, 4000, color='r', linestyles='dashed')
plt.legend(loc=0)
plt.xlabel('Iteration time')