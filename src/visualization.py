import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers, PillowWriter
import matplotlib.gridspec as gridspec
import numpy as np
import math

# personal func:
from models import BaseModel

# Constants
XLAB = 'time [s]'

# %% General


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


# %% Viz 3d

def viz_basedata(ax, col, basedata: np.ndarray, basemodel, frame: int):
    """draw frame"""
    # get all points
    points = basedata[frame, :, :]
    lines = []

    for count, line in enumerate(basemodel.connect()):
        x = [points[line[0], 0], points[line[1], 0]]
        y = [points[line[0], 1], points[line[1], 1]]
        z = [points[line[0], 2], points[line[1], 2]]

        if count > 0:
            line = ax.plot(x, y, z, col, label='_nolegend_')
        else:
            line = ax.plot(x, y, z, col,)

        lines.append(line[0])

    return ax, lines


def update_lines(frame: int, basedata: np.ndarray, basemodel,  lines: list):
    """
    update func for animation
    """
    points = basedata[frame, :, :]

    for line, pos in zip(lines, basemodel.connect()):
        x = [points[pos[0], 0], points[pos[1], 0]]
        y = [points[pos[0], 1], points[pos[1], 1]]
        z = [points[pos[0], 2], points[pos[1], 2]]

        line.set_data_3d(x, y, z)


def update_fig(frame: int, data: dict, basemodel, lines_dict: dict, extend=False):
    """
    update all elements
    """
    for key in data.keys():
        update_lines(frame, data[key], basemodel, lines_dict[key])

    if extend:
        sub_keys = ['sub1', 'sub2']
        for sub_key in sub_keys:
            x = lines_dict[f'{sub_key}_d']['t'][:frame]
            for key in lines_dict[sub_key].keys():
                y = lines_dict[f'{sub_key}_d'][key][:frame]
                lines_dict[sub_key][key].set_data(x, y)


def viz_all_models(frame: int, data: dict, basemodel, title=None, extend=False):
    """
    Draw all three models in one plot
    """

    if extend:
        fig = plt.figure('1', figsize=(8, 14))
        fig.clf()
        gridspec.GridSpec(4, 4)
        ax = plt.subplot2grid((4, 4), (0, 0), colspan=4,
                              rowspan=2, projection='3d')
    else:
        fig = plt.figure('1', figsize=(8, 8))
        fig.clf()
        ax = fig.gca(projection='3d')

    # init plot
    plt.title(title)
    lines_dic = {}
    _, lines_dic['vic'] = viz_basedata(ax, 'b', data['vic'], basemodel, frame)
    _, lines_dic['dec'] = viz_basedata(ax, 'r', data['dec'], basemodel, frame)
    _, lines_dic['imu'] = viz_basedata(ax, 'g', data['imu'], basemodel, frame)

    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])

    ax.set_zlim3d(
        [-4, 0]) if type(basemodel) == 'BaseModel' else ax.set_zlim3d([-4, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['Vicon', 'VideoPose3D', 'IMU'])

    if extend:
        plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=1)
        lines_dic['sub1_d'], lines_dic['sub1'] = plot_angle(
            basemodel.get_knee_angles, 'knee', data, 'left', lr=0, normal_mode=False, xlab=False)

        plt.subplot2grid((4, 4), (3, 0), colspan=4, rowspan=1)
        lines_dic['sub2_d'], lines_dic['sub2'] = plot_angle(
            basemodel.get_knee_angles, 'knee', data, 'right', lr=1, normal_mode=False, xlab=True)

    return fig, lines_dic

# %% make movie:


def make_movie(basemodel, data: dict, output: str, fps_fac=2, title=None, extend=True):
    """
    summarize funcions in movie, FuncAnimation will update all elements within plot
    """
    # init
    datal = len(data['vic'])
    frames = np.arange(0, datal, fps_fac)
    fps = int(100 / fps_fac)

    # init plot
    fig, lines_dic = viz_all_models(
        0, data, basemodel, title=title, extend=extend)

    # generate anim
    anim = FuncAnimation(fig, update_fig, frames=frames,
                         fargs=(data, basemodel, lines_dic, extend), interval=1000/fps)

    # save result
    if output.endswith('.mp4'):
        writer_cl = writers['ffmpeg']
        writer = writer_cl(fps=fps, metadata={})
        anim.save(output, writer=writer)
    else:
        anim.save(output, dpi=100, writer=PillowWriter(fps=fps))


# %% viz 2d


def plot_point(data: dict, point: int, coo=0):
    """Plot the coordinates of one point"""

    plt.figure(f'Point: {point}')
    plt.clf()
    plt.grid(0.25)
    datal = len(data['vic'])
    t = np.linspace(0, datal / 100, datal)

    for i, key in enumerate(data.keys()):
        plt.subplot(3, 1, i+1)
        plt.grid(0.25)
        loc_data = data[key][:, point, coo]
        loc_data = loc_data / np.max(loc_data)
        plt.plot(t, loc_data)
        plt.ylabel(key)

    plt.xlabel(XLAB)


def plot_angle(func, name, data: dict, cur_file: str, lr=0, normal_mode=True, xlab=True):
    """Plot the knee angle, lr = 1 -> left, lr = 0 -> right"""
    # init
    if normal_mode:
        plt.figure()
        plt.clf()
        #plt.subplot(2, 1, 1)

    fac = 180 / math.pi
    label = 'right' if lr == 0 else 'left'
    datal = len(data['vic'])
    t = np.linspace(0, datal / 100, datal)

    # plot
    plt.grid(0.25)
    plt.title(f'{label}-{name} angle') if normal_mode else None

    vic_y = func(data['vic'])[:, lr]*fac
    dec_y = func(data['dec'])[:, lr]*fac
    imu_y = func(data['imu'])[:, lr]*fac

    vic = plt.plot(t, vic_y, 'b')
    dec = plt.plot(t, dec_y, 'r')
    imu = plt.plot(t, imu_y, 'g')

    plt.xlabel('time [s]')
    #plt.legend(['Vicon', 'VideoPose3D', 'IMU']) if normal_mode else None
    plt.ylabel(f'{label}-{name} flexion [°]')

    plt.ylim([-50, 110])

    err_mode = False

    if err_mode:
        plt.subplot(2, 1, 2)
        plt.grid(0.25)

        mse_dec = np.sqrt((np.square(dec_y - vic_y)))
        mse_imu = np.sqrt((np.square(imu_y - vic_y)))

        plt.plot([0, 0], [0, 0], 'b')
        plt.plot(t, mse_dec, 'r')
        plt.plot(t, mse_imu, 'g')

        plt.xlabel(XLAB)
        plt.ylabel('Error[°]')
        #plt.legend(['Vicon', 'VideoPose3D', 'IMU'])
        plt.ylim([0, 75])

    if normal_mode:
        pass

    else:
        extend_data = {
            't': t.copy(),
            'vic': vic_y.copy(),
            'dec': dec_y.copy(),
            'imu': imu_y.copy()
        }
        plot_data = {
            'vic': vic[0],
            'dec': dec[0],
            'imu': imu[0]
        }

        return extend_data, plot_data
