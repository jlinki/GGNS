import os

import numpy as np
from matplotlib import pyplot as plt, animation as animation
from src.utils.visualization.visualization import png2gif


def animate_rollout_mesh_overlay(root_dir, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, triangles_grid_pred=None, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves the rollout of predicted and ground truth mesh using trisurf plots
    """
    if triangles_grid_pred is None:
        triangles_grid_pred = triangles_grid
    def one_frame_figure(i):
        fig = plt.figure() # if frame is needed: linewidth=1, edgecolor='black')
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(111, projection='3d')
        plot_one_frame(ax, i)
        return fig

    def plot_one_frame(ax,i):
        surf1 = ax.plot_trisurf(rollout_collider[stride * i][:, 0], rollout_collider[stride * i][:, 1],
                                np.zeros_like(rollout_collider[stride * i][:, 1]), triangles=triangles_collider,
                                color='blue', alpha=0.5, zorder=1)
        surf2 = ax.plot_trisurf(rollout_target[stride * i][:, 0], rollout_target[stride * i][:, 1],
                                np.zeros_like(rollout_target[stride * i][:, 1]), triangles=triangles_grid, color='red',
                                alpha=0.4, edgecolor='dimgrey', linewidth=0.5, label="Groundtruth", zorder=0)
        if isinstance(triangles_grid_pred, list):
            surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                    0.05 * np.ones_like(rollout_predicted[stride * i][:, 1]), triangles=triangles_grid_pred[stride*i],
                                    color='yellow',
                                    alpha=0.5, edgecolor='grey', linewidth=0.5, label="Prediction", zorder=2)
        else:
            surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                    0.05 * np.ones_like(rollout_predicted[stride * i][:, 1]), triangles=triangles_grid_pred,
                                    color='yellow',
                                    alpha=0.5, edgecolor='grey', linewidth=0.5, label="Prediction", zorder=2)
        #ax.set(xlim=(-1.00, 1.30), ylim=(-1.00, 1.30), zlim=(-1.0, 1.0))
        #ax.set(xlim=(-0.90, 1.00), ylim=(-0.7, 1.2), zlim=(-1.0, 1.0))
        #ax.set(xlim=(-0.90, 1.00), ylim=(-0.5, 1.4), zlim=(-1.0, 1.0))
        ax.set(xlim=(-1.00, 1.00), ylim=(-0.7, 1.1), zlim=(-1.0, 1.0))
        ax.azim = 270
        ax.elev = 90
        ax.axis('off')
        surf1._edgecolors2d = surf1._edgecolor3d
        surf1._facecolors2d = surf1._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d
        surf2._facecolors2d = surf2._facecolor3d
        surf3._edgecolors2d = surf3._edgecolor3d
        surf3._facecolors2d = surf3._facecolor3d
        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax.legend(loc=loc)#, fontsize=24)
        ax.set_title(f"t = {i*stride+1}")


    def animate(i):
        ax.clear()
        plot_one_frame(ax, i)


    plt.rcParams.update({'figure.constrained_layout.use': True})
    if save_animation:
        plt.rcParams.update({'axes.titlesize': 26})#, "font.family": "serif"})
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        for i in range(num_frames):
            fig = one_frame_figure(i)
            #subtitle = f"t = {i*stride+1}"
            #fig.text(0.5, 0.02, subtitle, ha='center', fontsize=26) # stripe
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'))
            plt.close()
        png2gif(num_frames, save_folder_dir)
    else:
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(111, projection='3d')
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=True)
        plt.show()


def animate_rollout_pcd_mesh_overlay(root_dir, rollout_predicted, rollout_pcd, rollout_collider, triangles_grid, triangles_collider, triangles_grid_pred=None, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves the rollout of predicted and ground truth mesh using trisurf plots
    """
    if triangles_grid_pred is None:
        triangles_grid_pred = triangles_grid
    def one_frame_figure(i):
        fig = plt.figure() # if frame is needed: linewidth=1, edgecolor='black')
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d')
        plot_one_frame(ax, i)
        return fig

    def plot_one_frame(ax,i):
        surf1 = ax.plot_trisurf(rollout_collider[stride * i][:, 0], rollout_collider[stride * i][:, 1],
                                np.zeros_like(rollout_collider[stride * i][:, 1]), triangles=triangles_collider,
                                color='cornflowerblue', alpha=1.0, zorder=1)
        surf2 = ax.scatter(rollout_pcd[i][:, 0], rollout_pcd[i][:, 1],
                                color='darkblue',
                                alpha=1.0, label="Pointcloud", zorder=10, s=20)
        if isinstance(triangles_grid_pred, list):
            surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                    -0.05 * np.ones_like(rollout_predicted[stride * i][:, 1]), triangles=triangles_grid_pred[stride*i],
                                    color='yellow',
                                    alpha=1.0, edgecolor='dimgrey', linewidth=1.5, label="Prediction", zorder=2)
        else:
            surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                    -0.05 * np.ones_like(rollout_predicted[stride * i][:, 1]), triangles=triangles_grid_pred,
                                    color='yellow',
                                    alpha=1.0, edgecolor='dimgrey', linewidth=1.5, label="Prediction", zorder=2)
        #ax.set(xlim=(-1.00, 1.30), ylim=(-1.00, 1.30), zlim=(-1.0, 1.0))
        #ax.set(xlim=(-0.90, 1.00), ylim=(-0.7, 1.2), zlim=(-1.0, 1.0))
        #ax.set(xlim=(-0.90, 1.00), ylim=(-0.5, 1.4), zlim=(-1.0, 1.0))
        ax.set(xlim=(-0.90, 0.90), ylim=(-0.5, 1.3), zlim=(-1.0, 1.0))
        ax.azim = 270
        ax.elev = 90
        ax.axis('off')
        surf1._edgecolors2d = surf1._edgecolor3d
        surf1._facecolors2d = surf1._facecolor3d
        #surf2._edgecolors2d = surf2._edgecolor3d
        #surf2._facecolors2d = surf2._facecolor3d
        surf3._edgecolors2d = surf3._edgecolor3d
        surf3._facecolors2d = surf3._facecolor3d
        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        #ax.legend(loc=loc)#, fontsize=24)
        #ax.set_title(f"t = {i*stride+1}")

    def animate(i):
        ax.clear()
        plot_one_frame(ax, i)


    plt.rcParams.update({'figure.constrained_layout.use': True})
    if save_animation:
        plt.rcParams.update({'axes.titlesize': 26})#, "font.family": "serif"})
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        for i in range(num_frames):
            fig = one_frame_figure(i)
            #subtitle = f"t = {i*stride+1}"
            #fig.text(0.5, 0.02, subtitle, ha='center', fontsize=26) # stripe
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'), dpi=200, transparent=True)
            plt.close()
        png2gif(num_frames, save_folder_dir)
    else:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d')
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()


def animate_rollout_export(root_dir, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, triangles_grid_pred=None, fig_number=0, loss="", stride=1, save_animation=False, ground_truth=False, zoom=False):
    """
    Animates or saves the rollout of predicted and ground truth mesh using trisurf plots
    """
    if triangles_grid_pred is None:
        triangles_grid_pred = triangles_grid
    def one_frame_figure(i):
        if zoom:
            fig = plt.figure(linewidth=15, edgecolor='red') # if frame is needed: linewidth=1, edgecolor='black')
        else:
            fig = plt.figure(linewidth=15, edgecolor='black')
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        ax.set_facecolor("black")
        ax.patch.set_edgecolor('black')
        plot_one_frame(ax, i)
        return fig

    def plot_one_frame(ax,i):
        surf1 = ax.plot_trisurf(rollout_collider[stride * i][:, 0], rollout_collider[stride * i][:, 1],
                                np.zeros_like(rollout_collider[stride * i][:, 1]), triangles=triangles_collider,
                                color='royalblue', alpha=1.0, zorder=1, edgecolor='royalblue', linewidth=0.1)
        if ground_truth:
            surf2 = ax.plot_trisurf(rollout_target[stride * i][:, 0], rollout_target[stride * i][:, 1],
                                    np.zeros_like(rollout_target[stride * i][:, 1]), triangles=triangles_grid, color=(0.8, 0.7, 0.0),
                                    alpha=1.0, edgecolor=(0.1, 0.1, 0.1), linewidth=0.3, label="Groundtruth", zorder=0)
        else:
            surf2 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                    np.zeros_like(rollout_target[stride * i][:, 1]), triangles=triangles_grid, color=(0.8, 0.7, 0.0),
                                    alpha=1.0, edgecolor=(0.1, 0.1, 0.1), linewidth=0.3, label="Prediction", zorder=0)
        if zoom:
            ax.set(xlim=(0.3, 1.30), ylim=(-0.8, 0.1), zlim=(-1.0, 1.0))
        else:
            ax.set(xlim=(-1.00, 1.00), ylim=(-0.75, 1.25), zlim=(-1.0, 1.0))
        ax.azim = 270
        ax.elev = 90
        ax.axis('off')
        surf1._edgecolors2d = surf1._edgecolor3d
        surf1._facecolors2d = surf1._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d
        surf2._facecolors2d = surf2._facecolor3d

    def animate(i):
        ax.clear()
        plot_one_frame(ax, i)

    plt.rcParams.update({'figure.constrained_layout.use': True})
    if save_animation:
        plt.rcParams.update({'axes.titlesize': 26})#, "font.family": "serif"})
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        for i in range(num_frames):
            fig = one_frame_figure(i)
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            if zoom:
                if i == num_frames-1:
                    plt.savefig(os.path.join(save_folder_dir, 'image_100' + '.png')) #, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor()
            else:
                plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'))
            plt.close()
        if not zoom:
            png2gif(num_frames, save_folder_dir)
            shape = [(260.0, 215.0), (495.0, 450.0)]
            from PIL import Image, ImageDraw
            import sys
            with Image.open(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png')) as im:
                draw = ImageDraw.Draw(im)
                draw.rectangle(shape, outline="red", width=4)
                im.save(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'))


    else:
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        ax.set_facecolor("black")
        ax.patch.set_edgecolor('black')
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()



def animate_rollout(rollout_predicted: list, rollout_target: list, rollout_collider: list, fig_number: int = 0, loss: int = "", stride: int = 1, save_animation: bool = False):
    """
    Animates or saves the rollout of predicted and ground truth mesh using scatter plots
    """
    num_frames = int(len(rollout_predicted)/stride)

    # create figure and axes objects
    plt.rcParams.update({'figure.constrained_layout.use': True})
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    def animate(i):
        ax.clear()
        ax.scatter(rollout_target[stride*i][:,0],rollout_target[stride*i][:,1], color='red', s=80, marker='+', linewidth=2.0, label="Groundtruth")
        ax.scatter(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1], s=80, color='blue', marker='x',
                   linewidth=2.0, label="Prediction")
        ax.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1], color='black', marker='.', linewidth=2.0, s=100)
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        ax.set_box_aspect(1)
        ax.axis('off')
        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax.legend(loc=loc, fontsize=26, frameon=True, framealpha=0.75)
        ax.set_title(f"Timestep: {i*stride+1}")

    # call the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=False)
    if save_animation:
        ani.save(os.path.join('./animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)), 'point_rollout.gif'), writer='pillow', fps=11, dpi=200)
    else:
        # show the plot
        plt.show()
