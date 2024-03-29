from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker


def plot_image(value_fn):
    # Values over all initial dealer scores and player scores
    # Given the policy, the value should be near zero everywhere except when the player has a score of 21
    dealer_scores = np.arange(1, 11, 1)  # 10
    player_scores = np.arange(11, 22, 1)  # 11
    V = np.zeros(shape=(len(dealer_scores), len(player_scores)))
    for d_idx, dealer_score in enumerate(dealer_scores):
        for p_idx, player_score in enumerate(player_scores):
            value = value_fn(dealer_score, player_score)
            V[d_idx][p_idx] = value
    fig, ax = plt.subplots()
    ax.imshow(V)
    ax.set_ylabel("Dealer initial showing")
    ax.yaxis.set_ticklabels(np.arange(0, 11, 1))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax.set_xlabel("Player sum")
    ax.xaxis.set_ticklabels(np.arange(10, 22, 1))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    plt.show()


def plot(value_fn):
    # Values over all initial dealer scores and player scores
    # Given the policy, the value should be near zero everywhere except when the player has a score of 21
    dealer_scores = np.arange(1, 11, 1)  # 10
    player_scores = np.arange(11, 22, 1)  # 11
    V = np.zeros(shape=(len(dealer_scores), len(player_scores)))
    for d_idx, dealer_score in enumerate(dealer_scores):
        for p_idx, player_score in enumerate(player_scores):
            value = value_fn(dealer_score, player_score)
            V[d_idx][p_idx] = value

    D, P = np.meshgrid(dealer_scores, player_scores)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(D, P, V.transpose())  # somehow we have to transpose here
    ax.set_xlabel("Dealer initial showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    plt.show()
