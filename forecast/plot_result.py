import matplotlib.pyplot as plt
import numpy as np


def get_moving_avgs(arr, window, convolution_mode):
    return (
        np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode,
        )
        / window
    )


# Smooth over a ... episode window
def plot_result(
    rewards_per_episode: list[float],
    losses_per_episode: list[float],
    rolling_length=20,
):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    axs[0].set_title("Episode Reward")
    reward_moving_average = get_moving_avgs(
        rewards_per_episode, rolling_length, "valid"
    )
    axs[0].plot(
        range(len(reward_moving_average)),
        reward_moving_average,
        label="Reward",
        color="green",
    )
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Training Errors")
    training_error_moving_average = get_moving_avgs(
        losses_per_episode, rolling_length, "same"
    )
    axs[1].plot(
        range(len(training_error_moving_average)),
        training_error_moving_average,
        label="Loss",
        color="red",
    )
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Total Loss")
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig("results/training_result_graph.png")
    plt.show()


if __name__ == "__main__":
    plot_result([10, 20, 30, 10, 10], [])
