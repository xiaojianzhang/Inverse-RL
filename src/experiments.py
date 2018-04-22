"""
Perform the experiments from the report.

Sheng Zhang
shengzhang@gatech.edu
"""

import time
from sys import stdout

import numpy as np
import matplotlib.pyplot as plt

import maxent
import deep_maxent
import value_iteration
from objectworld import Objectworld


def test_ow_once(grid_size, n_objects, n_colours, discrete, l1, l2, n_samples,
                 epochs, structure):
    """
    Test MaxEnt and DeepMaxEnt on a ow of size grid_size with the feature
    map feature_map with n_samples paths.

    grid_size: Grid size. int.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    discrete: Whether the features should be discrete. bool.
    l1: L1 regularisation. float.
    l2: L2 regularisation. float.
    n_samples: Number of paths to sample.
    epochs: Number of epochs to run MaxEnt with.
    structure: Neural network structure tuple, e.g. (3, 3) would be a
        3-layer neural network with assumed inputs.
    -> Expected value difference for MaxEnt, DeepMaxEnt
    """

    # Basic gist of what we're doing here: Get the reward function using our
    # different IRL methods, use those to get a policy, evaluate that policy
    # using the true reward, and then return the difference in expected values.

    # Setup parameters.
    wind = 0.3
    discount = 0.9
    learning_rate = 0.01
    trajectory_length = 3*grid_size

    # Make the objectworld and associated data.
    ow = Objectworld(grid_size, n_objects, n_colours, wind, discount)
    feature_matrix = ow.feature_matrix(discrete)
    ground_reward = np.array([ow.reward(i) for i in range(ow.n_states)])
    optimal_policy = value_iteration.find_policy(ow.n_states,
                                                 ow.n_actions,
                                                 ow.transition_probability,
                                                 ground_reward,
                                                 discount).argmax(axis=1)
    trajectories = ow.generate_trajectories(n_samples,
                                            trajectory_length,
                                            optimal_policy.take)
    p_start_state = (np.bincount(trajectories[:, 0, 0], minlength=ow.n_states) /
                     trajectories.shape[0])

    # True value.
    optimal_V = value_iteration.optimal_value(ow.n_states,
                                              ow.n_actions,
                                              ow.transition_probability,
                                              ground_reward, ow.discount)

    # MaxEnt reward; policy; value.
    # maxent_reward = deep_maxent.irl((feature_matrix.shape[1],),
    #                                 feature_matrix,
    #                                 ow.n_actions,
    #                                 ow.discount,
    #                                 ow.transition_probability,
    #                                 trajectories, epochs, learning_rate,
    #                                 l1=l1, l2=l2)
    print('MaxEnt reward; policy; value')
    maxent_reward = maxent.irl(feature_matrix,
                               ow.n_actions,
                               ow.discount,
                               ow.transition_probability,
                               trajectories,
                               epochs,
                               learning_rate)

    maxent_policy = value_iteration.find_policy(ow.n_states,
                                                ow.n_actions,
                                                ow.transition_probability,
                                                maxent_reward,
                                                discount).argmax(axis=1)
    maxent_V = value_iteration.value(maxent_policy,
                                     ow.n_states,
                                     ow.transition_probability,
                                     ground_reward,
                                     ow.discount)
    maxent_EVD = optimal_V.dot(p_start_state) - maxent_V.dot(p_start_state)

    # DeepMaxEnt reward; policy; value.
    print('DeepMaxEnt reward; policy; value')
    deep_learning_rate = 0.005 # For the 32 x 32 experiments.
    deep_maxent_reward = deep_maxent.irl((feature_matrix.shape[1],)+structure,
                                         feature_matrix,
                                         ow.n_actions,
                                         ow.discount,
                                         ow.transition_probability,
                                         trajectories, epochs,
                                         deep_learning_rate,
                                         l1=l1, l2=l2)

    deep_maxent_policy = value_iteration.find_policy(ow.n_states,
                                                     ow.n_actions,
                                                     ow.transition_probability,
                                                     deep_maxent_reward,
                                                     discount).argmax(axis=1)
    deep_maxent_V = value_iteration.value(deep_maxent_policy,
                                          ow.n_states,
                                          ow.transition_probability,
                                          ground_reward,
                                          ow.discount)

    deep_maxent_EVD = (optimal_V.dot(p_start_state) -
                       deep_maxent_V.dot(p_start_state))

    plt.subplot(3, 3, 1)
    plt.pcolor(ground_reward.reshape((grid_size, grid_size)))
    plt.title("Groundtruth reward")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 2)
    plt.pcolor(maxent_reward.reshape((grid_size, grid_size)))
    plt.title("MaxEnt reward")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 3)
    plt.pcolor(deep_maxent_reward.reshape((grid_size, grid_size)))
    plt.title("DeepMaxEnt reward")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)

    plt.subplot(3, 3, 4)
    plt.pcolor(optimal_policy.reshape((grid_size, grid_size)), vmin=0, vmax=3)
    plt.title("Optimal policy")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 5)
    plt.pcolor(maxent_policy.reshape((grid_size, grid_size)), vmin=0, vmax=3)
    plt.title("MaxEnt policy")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 6)
    plt.pcolor(deep_maxent_policy.reshape((grid_size, grid_size)),
               vmin=0, vmax=3)
    plt.title("DeepMaxEnt policy")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)

    plt.subplot(3, 3, 7)
    plt.pcolor(optimal_V.reshape((grid_size, grid_size)))
    plt.title("Optimal value")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 8)
    plt.pcolor(maxent_V.reshape((grid_size, grid_size)))
    plt.title("MaxEnt value")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.subplot(3, 3, 9)
    plt.pcolor(deep_maxent_V.reshape((grid_size, grid_size)))
    plt.title("DeepMaxEnt value")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
        bottom=False, top=False, left=False, right=False, labelright=False)
    plt.savefig("{}_{}_{}_{}_{}_{}_{}_{}_{}_objectworld_{}.png".format(
        grid_size, n_objects, n_colours, discrete, n_samples, epochs, structure,
        l1, l2, np.random.randint(10000000)))

    return maxent_EVD, deep_maxent_EVD


def test_ow_over_samples(grid_size, n_objects, n_colours, discrete, l1, l2,
                         epochs, structure, n):
    """
    Test MaxEnt and DeepMaxEnt on an objectworld with different numbers of paths.

    grid_size: Grid size. int.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    discrete: Whether the features should be discrete. bool.
    feature_map: Which feature map to use. String in {ident, coord, proxi}.
    l1: L1 regularisation. float.
    l2: L2 regularisation. float.
    epochs: MaxEnt iterations. int.
    structure: Neural network structure tuple, e.g. (3, 3) would be a
        3-layer neural network with assumed inputs.
    n: Iterations. int.
    -> (MaxEnt [(n_samples, mean expected value difference, stdev)],
        DeepMaxEnt [(n_samples, mean expected value difference, stdev)]),
       raw data (maxent_data, deep_maxent_data)
    """

    maxent_data = []
    deep_maxent_data = []
    for n_samples in [256,128,64,32,16]:
        t = time.time()
        maxent_EVDs = []
        deep_maxent_EVDs = []
        for i in range(n):
            print("{}: {}/{}".format(n_samples, i+1, n))
            maxent_EVD, deep_maxent_EVD = test_ow_once(grid_size, n_objects, n_colours, discrete, l1, l2, n_samples, epochs, structure)
            maxent_EVDs.append(maxent_EVD)
            deep_maxent_EVDs.append(deep_maxent_EVD)
            print(maxent_EVD, deep_maxent_EVD)
            stdout.flush()
        maxent_data.append((n_samples, np.mean(maxent_EVDs), np.median(maxent_EVDs), np.std(maxent_EVDs)))
        deep_maxent_data.append((n_samples, np.mean(deep_maxent_EVDs), np.median(deep_maxent_EVDs), np.std(deep_maxent_EVDs)))
        print("{} samples and {} iterations (took {}s)".format(n_samples, n, time.time() - t))
        print("MaxEnt:", maxent_data)
        print("DeepMaxEnt:", deep_maxent_data)
    return maxent_data, deep_maxent_data

if __name__ == '__main__':
    # Tests the 16 x 16 objectworld.
    print(test_ow_over_samples(16, 25, 2, False, 0, 0, 300, (3, 3), 5))
