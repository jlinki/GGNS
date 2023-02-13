import os
import imageio
import numpy as np
from matplotlib import pyplot as plt


def plot_loss_curve(rollout_loss: list, rollout_time_difference_loss: list = None, save_animation: bool = False):
    """
    Simple plotting function for loss curves
    Args:
        rollout_loss: list containing the losses over time
        rollout_time_difference_loss: list containing the loss between two consecutive data samples over time
        save_animation: only display if not animations are saved
    """
    t = np.arange(len(rollout_loss))
    fig, ax = plt.subplots()
    if rollout_time_difference_loss is not None:
        ax.plot(t, np.asarray(rollout_loss), color='red', label='loss prediction')
        ax.plot(t, np.asarray(rollout_time_difference_loss), color='green', label='time difference loss')
        ax.legend()
    else:
        ax.plot(t, np.asarray(rollout_loss))
    ax.set(xlabel='time (0.01s)', ylabel='MSE loss')
    if not save_animation:
        plt.show()


def plot_velocity_curve(vel_predicted: list, vel_groundtruth: list, title: str, save_animation: bool = False):
    """
    Simple plotting function for loss curves
    Args:
        vel_predicted: list containing the predicted velocities over time
        vel_groundtruth: list containing the ground truth velocities over time samples over time
        title: title of figure
        save_animation: only display if not animations are saved
    """
    t = np.arange(len(vel_predicted))
    fig, ax = plt.subplots()
    if vel_groundtruth is not None:
        ax.plot(t, np.asarray(vel_predicted), color='red', label='predicted velocity')
        ax.plot(t, np.asarray(vel_groundtruth), color='green', label='ground truth velocity')
        ax.legend()
    else:
        ax.plot(t, np.asarray(vel_predicted))
    ax.set(xlabel='time (0.01s)', ylabel='MSE loss')
    ax.set_title(title)
    if not save_animation:
        plt.show()


def png2gif(num_frames: int, dir: str):
    """
    loads a number of png images and converts them into gif
    Args:
        num_frames: number of png images to use
        dir: directory to save under
    """
    filenames = []
    for i in range(num_frames):
        filenames.append(os.path.join(dir, 'image_' + str(i).zfill(3) + '.png'))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(dir, 'rollout.gif'), images, duration=0.08)
