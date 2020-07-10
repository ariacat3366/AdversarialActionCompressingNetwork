import numpy as np
import matplotlib.pyplot as plt
import math


def norm2d(x, y, var_x=1., var_y=1., num_sample=1000, pi=0.):

    result = np.zeros((num_sample, 2))
    sample_x = np.random.normal(x, var_x, num_sample)
    sample_y = np.random.normal(y, var_y, num_sample)
    cos = math.cos(pi)
    sin = math.sin(pi)
    replaced_x = sample_x * cos - sample_y * sin
    replaced_y = sample_x * sin + sample_y * cos

    return np.array(list(zip(replaced_x, replaced_y)))


def multi_norm2d(radius=0.5, num_dist=10, num_sample=5000):

    result = np.zeros((num_sample, 2))
    for i in range(num_dist):
        pi = math.pi * 2 * i / num_dist
        num_each_sample = num_sample // num_dist
        dist = norm2d(x=radius,
                      y=0,
                      var_x=0.2,
                      var_y=0.05,
                      num_sample=num_each_sample,
                      pi=pi)
        result[i * num_each_sample:(i + 1) * num_each_sample] = dist
    return result


if __name__ == "__main__":
    sample = multi_norm2d()
    plt.figure(figsize=(5, 5))
    plt.scatter(sample[:, 0], sample[:, 1], marker='.')
    plt.savefig("../log/figure/test_multi_norm2d.png", format="png", dpi=100)