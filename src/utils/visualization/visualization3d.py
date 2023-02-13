import os

import numpy as np
from matplotlib import pyplot as plt, animation as animation

from src.utils.visualization.visualization import png2gif


def animate_rollout3d(rollout_predicted, rollout_target, rollout_collider, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves the rollout of predicted and ground truth 3D mesh using scatter plots
    """
    num_frames = int(len(rollout_predicted)/stride)

    # create figure and axes objects
    plt.rcParams.update({'figure.constrained_layout.use': True})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.set_size_inches(5, 5)

    def animate(i):
        ax.clear()
        ax.set(xlim=(-0.05, 0.05), ylim=(-0.025, 0.025), zlim=(0.05, 0.15))
        ax.scatter(rollout_target[stride*i][:,0],rollout_target[stride*i][:,1], rollout_target[stride*i][:,2], color='red', s=40, marker='+', linewidth=1.0, label="Groundtruth", zorder=10)
        ax.scatter(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1], rollout_predicted[stride * i][:, 2], s=40, color='blue', marker='x',
                   linewidth=1.0, label="Prediction", zorder=100)
        ax.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1],rollout_collider[stride*i][:,2], color='black', marker='+', linewidth=2.0, s=400, label="Gripper", zorder=1000)

        ax.axis('off')
        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax.legend(loc=loc, fontsize=16, frameon=True, framealpha=0.75)
        ax.set_title(f"Timestep: {i*stride+1}")
        ax.azim = 225
        ax.elev = 45

    # call the animation
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50)
    if save_animation:
        ani.save(os.path.join('./animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)), 'point_rollout.gif'), writer='pillow', fps=11, dpi=200)
    else:
        # show the plot
        plt.show()


def animate_rollout3d_mesh_side_by_side(dir, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves side by side plots of the rollout of predicted and ground truth 3D mesh using trisurf plots for tissue manipulation dataset
    """
    def one_frame_figure(i):
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plot_one_frame(ax, ax2, i)
        return fig


    def plot_one_frame(ax, ax2, i, azim=225, elev=45):
        ##### Plot Groundtruth #####
        ax.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1],rollout_collider[stride*i][:,2], color='black', marker='+', linewidth=2.0, s=300, label="Gripper")
        surf2 = ax.plot_trisurf(rollout_target[stride * i][:, 0], rollout_target[stride * i][:, 1],
                                rollout_target[stride * i][:, 2], triangles=triangles_grid, color='yellow',
                                alpha=1.0, edgecolor='dimgrey', linewidth=0.5, label="Groundtruth", zorder=0)

        ax.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.9, 0.9))
        ax.azim = azim
        ax.elev = elev
        surf2._edgecolors2d = surf2._edgecolor3d
        surf2._facecolors2d = surf2._facecolor3d

        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax.axis('off')
        ax.set_title(r"$\bf{Groundtruth}$" + f" \n Timestep: {i*stride+1}")


        ##### Plot Prediction #####
        ax2.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1],rollout_collider[stride*i][:,2], color='black', marker='+', linewidth=2.0, s=300, label="Gripper")
        surf3 = ax2.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                rollout_predicted[stride * i][:, 2], triangles=triangles_grid,
                                color='yellow',
                                alpha=1.0, edgecolor='dimgrey', linewidth=0.5, label="Prediction", zorder=2)
        ax2.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.9, 0.9))
        ax2.azim = azim
        ax2.elev = elev
        surf3._edgecolors2d = surf3._edgecolor3d
        surf3._facecolors2d = surf3._facecolor3d

        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax2.axis('off')
        ax2.set_title(r"$\bf{Prediction}$" + " \n")

    def animate(i):
        ax.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        plot_one_frame(ax, ax2, i)
        plot_one_frame(ax3, ax4, i, -45, 45)


    plt.rcParams.update({'figure.constrained_layout.use': True})
    if save_animation:
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        save_folder_dir = os.path.join(dir, 'animations', 'side_by_side_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        for i in range(num_frames):
            fig = one_frame_figure(i)
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'))
            plt.close()
        png2gif(num_frames, save_folder_dir)
    else:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        num_frames = int(len(rollout_predicted) / stride)
        ax = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()


def animate_rollout3d_mesh_overlay(root_dir, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves rollout of predicted and ground truth 3D mesh using trisurf plots
    """

    def animate(i):
        ax.clear()

        surf2 = ax.plot_trisurf(rollout_target[stride * i][:, 0], rollout_target[stride * i][:, 1],
                                rollout_target[stride * i][:, 2], triangles=triangles_grid, color='red',
                                alpha=1.0, edgecolor='dimgrey', linewidth=0.5, label="Groundtruth", zorder=10)
        surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                rollout_predicted[stride * i][:, 2], triangles=triangles_grid,
                                color='yellow',
                                alpha=0.6, edgecolor='grey', linewidth=0.5, label="Prediction", zorder=100)
        ax.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1],rollout_collider[stride*i][:,2], color='black', marker='+', linewidth=3.0, label="Gripper", zorder=1000, s=300)
        # surf1 = ax.plot_trisurf(rollout_collider[stride * i][:, 0], rollout_collider[stride * i][:, 1],
        #                         rollout_collider[stride * i][:, 2], triangles=triangles_collider, color='darkgrey',
        #                         alpha=1.0, edgecolor='black', linewidth=0.5, label="Collider", zorder=10)
        ax.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.9, 0.9))
        ax.axis('off')
        # surf1._edgecolors2d = surf1._edgecolor3d
        # surf1._facecolors2d = surf1._facecolor3d
        surf2._edgecolors2d = surf2._edgecolor3d
        surf2._facecolors2d = surf2._facecolor3d
        surf3._edgecolors2d = surf3._edgecolor3d
        surf3._facecolors2d = surf3._facecolor3d
        if np.mean(rollout_collider[0][:, 0]) > 0:
            loc = 'upper left'
        else:
            loc = 'upper right'
        ax.legend(loc='upper left', fontsize=10)
        #ax.azim = -180
        #ax.elev = 20
        ax.set_title(f"Timestep: {i*stride+1}")

    plt.rcParams.update({'figure.constrained_layout.use': True})

    if save_animation:
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        for i in range(num_frames):
            animate(i)
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'), dpi=80)
        png2gif(num_frames, save_folder_dir)

    else:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()


def animate_rollout3d_tube_mesh_overlay(root_dir, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, fig_number=0, loss="", stride=1, save_animation=False):
    """
    Animates or saves rollout of predicted and ground truth 3D mesh using trisurf plots for cavity deformation dataset
    """

    def animate(i):
        ax.clear()

        surf2 = ax.plot_trisurf(rollout_target[stride * i][:, 0], rollout_target[stride * i][:, 1],
                                rollout_target[stride * i][:, 2], triangles=triangles_grid, color='red',
                                alpha=1.0, edgecolor='dimgrey', linewidth=0.5, label="Groundtruth", zorder=10)
        surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                rollout_predicted[stride * i][:, 2], triangles=triangles_grid,
                                color='yellow',
                                alpha=0.6, edgecolor='grey', linewidth=0.5, label="Prediction", zorder=100)
        #ax.scatter(rollout_collider[stride*i][:,0],rollout_collider[stride*i][:,1],rollout_collider[stride*i][:,2], color='black', marker='+', linewidth=3.0, label="Gripper", zorder=1000)
        surf1 = ax.plot_trisurf(rollout_collider[stride * i][:, 0], rollout_collider[stride * i][:, 1],
                                rollout_collider[stride * i][:, 2], triangles=triangles_collider, color='darkgrey',
                                alpha=1.0, edgecolor='black', linewidth=0.5, label="Collider", zorder=10)
        ax.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.9, 0.9))
        ax.set(xlim=(-0.35, 0.35), ylim=(-0.45, 0.45), zlim=(-0.3, 0.3))
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
        ax.legend(loc='upper left', fontsize=10)
        ax.azim = -170
        ax.elev = 10
        ax.set_title(f"Timestep: {i*stride+1}")

    plt.rcParams.update({'figure.constrained_layout.use': True})

    if save_animation:
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        for i in range(num_frames):
            animate(i)
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'), dpi=80)
        png2gif(num_frames, save_folder_dir)

    else:
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()



def animate_rollout3d_scene(root_dir, input_dataset, data_dict, rollout_predicted, rollout_target, rollout_collider, triangles_grid, triangles_collider, fig_number=0, loss="", stride=1, save_animation=False, ground_truth=False, zoom=False):
    """
    Animates or saves complete scene rollout of predicted and ground truth 3D mesh using trisurf plots
    """

    if 'tube' in input_dataset:
        tube_mesh_positions = data_dict["tube_mesh_positions"]
        tube_mesh_triangles = data_dict["tube_mesh_triangles"]
        gripper_mesh_positions = data_dict["gripper_mesh_positions"]
        gripper_mesh_triangles = data_dict["gripper_mesh_triangles"]
        panda_mesh_positions = data_dict["panda_mesh_positions"]
        panda_mesh_triangles = data_dict["panda_mesh_triangles"]
        point_cloud_positions = data_dict["point_cloud_positions"]
        split_positions = int(gripper_mesh_positions[0].shape[0]*0.5)
        split_triangles = int(gripper_mesh_triangles.shape[0]*0.5)
        gripper1_mesh_positions = []
        for gripper_mesh_position in gripper_mesh_positions:
            gripper1_mesh_positions.append(gripper_mesh_position[0:split_positions,:])
        gripper1_mesh_triangles = gripper_mesh_triangles[0:split_triangles,:]
        gripper2_mesh_positions = []
        for gripper_mesh_position in gripper_mesh_positions:
            gripper2_mesh_positions.append(gripper_mesh_position[split_positions:,:])
        gripper2_mesh_triangles = gripper_mesh_triangles[split_triangles:,:]
        gripper2_mesh_triangles = gripper2_mesh_triangles - split_positions

        def animate(i):
            ax.clear()
            #ax.patch.set_edgecolor('black')
            #ax.set_facecolor("black")
            #(0.8, 0.7, 0.0)
            surf1 = ax.plot_trisurf(gripper1_mesh_positions[stride * i][:, 0], gripper1_mesh_positions[stride * i][:, 1],
                                     gripper1_mesh_positions[stride * i][:, 2], triangles=gripper1_mesh_triangles, color='crimson',
                                     alpha=1.0, edgecolor='crimson', linewidth=0.1, label="Gripper1", zorder=100)
            surf3 = ax.plot_trisurf(gripper2_mesh_positions[stride * i][:, 0], gripper2_mesh_positions[stride * i][:, 1],
                                     gripper2_mesh_positions[stride * i][:, 2], triangles=gripper2_mesh_triangles, color='crimson',
                                     alpha=1.0, edgecolor='crimson', linewidth=0.1, label="Gripper2", zorder=2000)
            surf2 = ax.plot_trisurf(panda_mesh_positions[stride * i][:, 0], panda_mesh_positions[stride * i][:, 1],
                                    panda_mesh_positions[stride * i][:, 2], triangles=panda_mesh_triangles, color='darkgrey',
                                    alpha=1.0, edgecolor='grey', linewidth=0.1, label="Panda", zorder=10)
            if ground_truth:
                surf4 = ax.plot_trisurf(tube_mesh_positions[stride * i][:, 0], tube_mesh_positions[stride * i][:, 1],
                                        tube_mesh_positions[stride * i][:, 2], triangles=tube_mesh_triangles,
                                        color=(0.85, 0.75, 0.0),
                                        alpha=1.0, edgecolor=(0.15, 0.15, 0.15), linewidth=0.1, label="Predicted Mesh", zorder=1000)
            else:
                surf4 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                        rollout_predicted[stride * i][:, 2], triangles=tube_mesh_triangles,
                                        color=(0.85, 0.75, 0.0),
                                        alpha=1.0, edgecolor=(0.15, 0.15, 0.15), linewidth=0.1, label="Predicted Mesh", zorder=1000)

            if zoom:
                ax.set(xlim=(-0.35, 0.35), ylim=(-0.45, 0.45), zlim=(-0.3, 0.3))
            else:
                ax.set(xlim=(-0.35, 0.35), ylim=(-0.45, 0.45), zlim=(-0.3, 0.3))
            ax.axis('off')
            surf1._edgecolors2d = surf1._edgecolor3d
            surf1._facecolors2d = surf1._facecolor3d
            surf2._edgecolors2d = surf2._edgecolor3d
            surf2._facecolors2d = surf2._facecolor3d
            surf3._edgecolors2d = surf3._edgecolor3d
            surf3._facecolors2d = surf3._facecolor3d
            surf4._edgecolors2d = surf4._edgecolor3d
            surf4._facecolors2d = surf4._facecolor3d
            if np.mean(rollout_collider[0][:, 0]) > 0:
                loc = 'upper left'
            else:
                loc = 'upper right'
            #ax.legend(loc='upper left', fontsize=10)
            ax.azim = -175
            ax.elev = 10
            #ax.set_title(f"Timestep: {i*stride+1}")

    elif 'tissue' in input_dataset:
        tissue_mesh_positions = data_dict["tissue_mesh_positions"]
        tissue_mesh_triangles = data_dict["tissue_mesh_triangles"]
        gripper_mesh_positions = data_dict["gripper_mesh_positions"]
        gripper_mesh_triangles = data_dict["gripper_mesh_triangles"]
        liver_mesh_positions = data_dict["liver_mesh_positions"]
        liver_mesh_triangles = data_dict["liver_mesh_triangles"]
        point_cloud_positions = data_dict["point_cloud_positions"]

        def animate(i):
            ax.clear()
            surf1 = ax.plot_trisurf(gripper_mesh_positions[stride * i][:, 0], gripper_mesh_positions[stride * i][:, 1],
                                     gripper_mesh_positions[stride * i][:, 2], triangles=gripper_mesh_triangles, color='darkgrey',
                                     alpha=1.0, edgecolor='grey', linewidth=0.1, label="Gripper", zorder=1000)
            # surf2 = ax.plot_trisurf(liver_mesh_positions[:, 0], liver_mesh_positions[:, 1],
            #                         liver_mesh_positions[:, 2], triangles=liver_mesh_triangles, color='red',
            #                         alpha=1.0, edgecolor='black', linewidth=0.5, label="Liver")
            if ground_truth:
                surf3 = ax.plot_trisurf(tissue_mesh_positions[stride * i][:, 0], tissue_mesh_positions[stride * i][:, 1],
                                                    rollout_predicted[stride * i][:, 2], triangles=tissue_mesh_triangles,
                                                    color=(0.95, 0.85, 0.0),
                                                    alpha=1.0, edgecolor=(0.0, 0.0, 0.0), linewidth=0.3, label="Predicted Mesh")
            else:
                surf3 = ax.plot_trisurf(rollout_predicted[stride * i][:, 0], rollout_predicted[stride * i][:, 1],
                                        rollout_predicted[stride * i][:, 2], triangles=tissue_mesh_triangles,
                                        color=(0.95, 0.85, 0.0),
                                        alpha=1.0, edgecolor='black', linewidth=0.3, label="Predicted Mesh")

            if zoom:
                ax.set(xlim=(-0.5, 0.85), ylim=(-0.75, 0.6), zlim=(-0.52, 0.52))
            else:
                ax.set(xlim=(-0.8, 1.1), ylim=(-0.6, 1.3), zlim=(-0.6, 0.8))
            ax.axis('off')
            surf1._edgecolors2d = surf1._edgecolor3d
            surf1._facecolors2d = surf1._facecolor3d
            #surf2._edgecolors2d = surf2._edgecolor3d
            #surf2._facecolors2d = surf2._facecolor3d
            surf3._edgecolors2d = surf3._edgecolor3d
            surf3._facecolors2d = surf3._facecolor3d
            ax.azim = -10
            ax.elev = 0

            if np.mean(rollout_collider[0][:, 0]) > 0:
                loc = 'upper left'
            else:
                loc = 'upper right'

    plt.rcParams.update({'figure.constrained_layout.use': True})

    if save_animation:
        stride = 1
        num_frames = int(len(rollout_predicted) / stride)
        save_folder_dir = os.path.join(root_dir, 'animations', 'overlay_animation_rollout', str(fig_number) + "_loss_" + str(int(loss)))
        fig = plt.figure(linewidth=15, edgecolor='black')
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False, facecolor='black')
        ax.set_facecolor("black")
        ax.patch.set_edgecolor('black')
        for i in range(num_frames):
            animate(i)
            if not os.path.exists(save_folder_dir):
                os.makedirs(save_folder_dir)
            plt.savefig(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'), dpi=80)
            if i == num_frames-1:
                plt.savefig(os.path.join(save_folder_dir, 'image_' + '100_crop.png'), dpi=160)
                from PIL import Image, ImageDraw
                with Image.open(os.path.join(save_folder_dir, 'image_' + '100_crop.png')) as img:
                    if 'tube' in input_dataset:
                        area = (400, 0, 1200, 800)
                        shape = [(0.0, 0.0), (800.0, 800.0)]
                    else:
                        area = (400, 400, 1200, 1200)
                        shape = [(0.0, 0.0), (800.0, 800.0)]
                    cropped_img = img.crop(area)
                    draw = ImageDraw.Draw(cropped_img)
                    draw.rectangle(shape, outline="red", width=6)
                    cropped_img.save(os.path.join(save_folder_dir, 'image_' + '100') + '.png')
        png2gif(num_frames, save_folder_dir)

        if 'tube' in input_dataset:
            shape = [(200.0, 0.0), (600.0, 400.0)]
        else:
            shape = [(200.0, 200.0), (600.0, 600.0)]
        from PIL import Image, ImageDraw
        import sys
        with Image.open(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png')) as im:
            draw = ImageDraw.Draw(im)
            draw.rectangle(shape, outline="red", width=6)
            im.save(os.path.join(save_folder_dir, 'image_' + str(i).zfill(3) + '.png'))

    else:
        fig = plt.figure(linewidth=15, edgecolor='black')
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False, facecolor='black')
        ax.set_facecolor("black")
        ax.patch.set_edgecolor('black')
        num_frames = int(len(rollout_predicted) / stride)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=False)
        plt.show()
