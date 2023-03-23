# %%
# we need 3 functions databunch to some important angles
import math
import numpy as np
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

# personal includes
from models import BaseModel, BaseExtModel, DecModel, VIC_KEYS
from visualization import viz_all_models, plot_point, plot_angle, make_movie

# Constants

BASEKEYS = [
    'pan_r',
    'knee_r',
    'foot_r',
    'pan_l',
    'knee_l',
    'foot_l',
]

EXTKEYS = BASEKEYS + ['hip_up']


# %% Read the Viondata CSV


def read_vicdata(filename: str, header=3, base=True):
    """
    - detectron files are csv files
    - the databunch contains 16 3dim positions for all frames
    """
    # read the file
    vicondata = pd.read_csv(filename, header=header,
                            dtype=np.float64, skiprows=1)
    # transform to basedata
    return vicdata_2_basedata(vicondata, base=base)


def vicdata_2_basedata(vicdata, take=2, istart=0, base=True):
    """from the 16 points available we only need the first 7"""
    # dim is (frames x points x coordinates)
    # init
    keys = vicdata.keys()[2:]  # omit the first two!
    data_len = len(vicdata[keys[1]][istart*take::take])
    basedata = np.zeros([data_len, 8, 3])

    indexes = [
        [VIC_KEYS['LASI'], VIC_KEYS['RASI']],  # SACR (LPSI + RPSI) / 2
        [VIC_KEYS['RASI']],  # RASI
        [VIC_KEYS['RKNE']],  # RKNE
        [VIC_KEYS['RANK']],  # RANK[13], RHEE[14]
        [VIC_KEYS['LASI']],  # LASI
        [VIC_KEYS['LKNE']],  # LKNE
        [VIC_KEYS['LANK']],  # LANK[7], LHEE[8]
        [VIC_KEYS['RPSI']],
    ]

    for i, index in enumerate(indexes):

        # build mean
        if len(index) == 2:
            idxr = range(3*index[0], 3*index[0]+3)
            loc_data1 = np.array(vicdata[keys[idxr]][istart*take::take])
            idxr = range(3*index[1], 3*index[1]+3)
            loc_data2 = np.array(vicdata[keys[idxr]][istart*take::take])
            loc_data = (loc_data1 + loc_data2) * 0.5

        # simply take
        else:
            idxr = range(3*index[0], 3*index[0]+3)
            loc_data = np.array(vicdata[keys[idxr]][istart*take::take])

        basedata[:, i, :] = loc_data

    # remove last frame if unequal to fit dec data
    if len(basedata) % take == 1:
        basedata = basedata[:-(take-1), :, :]
    # remove nans and zeros
    basedata = remove_nans(basedata)
    basedata = remove_nans(basedata, 'reversed')
    # take new center
    basedata[:, 0, :] = (basedata[:, 1, :] + basedata[:, 4, :]) * 0.5
    # center around base
    basedata = center_basedata(basedata)
    # add z-axis if not base mode
    if base:
        basedata = basedata[:, :7, :]
    else:
        basedata = vic_z_axis(basedata)

    return basedata


def vic_z_axis(basedata):
    """
    extend the vicon data with the corresponding z -axis
    we will overwrite the last point
    """
    sacrum_all = np.copy(basedata[:, 0, :])
    rpsi_all = np.copy(basedata[:, 7, :])
    lasi_all = np.copy(basedata[:, 4, :])
    rasi_all = np.copy(basedata[:, 1, :])

    y_all = [(lasi - rasi) / np.linalg.norm(lasi-rasi)
             for lasi, rasi in zip(lasi_all, rasi_all)]

    sec_dir = [rpsi - sacrum for rpsi, sacrum in zip(rpsi_all, sacrum_all)]

    z_all = [np.cross(y, sec) for y, sec in zip(y_all, sec_dir)]

    basedata[:, 7, :] = z_all

    return basedata


def remove_nans(basedata: np.ndarray, mode='forward'):
    """take the previous input if nan occurs"""

    if mode == 'forward':
        def mid_range(x): return range(x)

    else:
        def mid_range(x): return reversed(range(x))

    # local check function
    def check(x): return np.isnan(x) or x == 0

    shape_it = basedata.shape

    # go trough all points
    for point in range(shape_it[1]):
        xyz = np.zeros([3])

        # go over each line
        for row in mid_range(shape_it[0]):

            # go over x, y, z
            for coo in range(shape_it[2]):

                cur_coo = basedata[row, point, coo]

                if check(cur_coo):
                    basedata[row, point, coo] = xyz[coo]
                else:
                    xyz[coo] = cur_coo

    return basedata


def center_basedata(basedata):
    """substract the center from the rest to keep it centered around (0,0,0)"""
    shape_it = basedata.shape
    center = basedata[:, 0, :].copy()

    for point in range(shape_it[1]):
        basedata[:, point, :] -= center

    basedata = basedata / np.max(basedata)

    return basedata

# %% Read the VideoPose3D (Detectron2) generated data


def read_decdata(filename: str, base=True):
    """
    - detectron files are json files
    - the databunch contains 18 3dim positions for all frames
    """
    # read the file
    with open(filename, 'r') as f:
        decdata = json.load(f)

    # transform to basedata
    return decdata_2_basedata(decdata, base=base)


def decdata_2_basedata(decdata, istart=0, key='Reconstruction', base=True):
    """from the 16 points available we only need the first {7,8} depending on base mode"""
    # dim is (frames x points x coordinates)
    # refactor data to a numpy array
    endkey = 7 if base else 8

    basedata = np.array(decdata[key])
    basedata = basedata[istart:, :endkey, :]
    # center around base
    basedata = center_basedata(basedata)
    return basedata


# %% Read the Internal Measurement Unit (IMU) json

def read_imudata(filename, base=True):
    """
    - imu files are json files
    - they have the _imu ending
    - they contain a dict with positions (left knee, ...)
    """
    with open(filename, 'r') as f:
        imudata = json.load(f)

    return imudata_2_basedata(imudata, base=base)


def imudata_2_basedata(imu_data, base=True):
    """
    we will got from the foot_l, foot_r mode to the (frames x points x coordinates) view
    export the base keys if base mode else extkeys
    """
    # generic keys
    key_list = BASEKEYS if base else EXTKEYS

    data_len = len(imu_data[key_list[0]])
    basedata = np.zeros((data_len, len(key_list) + 1, 3))
    # evelate the z coordinate
    basedata[:, 0, 2] = np.ones((data_len)) * 2

    # fill the empty array
    for i, key in enumerate(key_list):
        loc_data = np.array(imu_data[key]).reshape([data_len, 3])
        basedata[:, i+1, :] = loc_data

    basedata = center_basedata(basedata)
    basedata = realign(basedata, 2)

    # swap x and y coordinates
    x_old = np.copy(basedata[:, :, 0])
    y_old = np.copy(basedata[:, :, 1])
    basedata[:, :, 0] = -y_old
    basedata[:, :, 1] = -x_old

    return basedata


def realign(basedata: np.ndarray, factor):
    """interpolate to new Hz number"""
    size_it = basedata.shape
    x_cur = np.array(list(range(size_it[0])))
    x_new = np.arange(0, size_it[0], 1 / factor)
    basedata_new = np.zeros([len(x_new), size_it[1], size_it[2]])

    for point in range(size_it[1]):
        for coo in range(size_it[2]):
            basedata_new[:, point, coo] = np.interp(
                x_new, x_cur, basedata[:, point, coo])

    return basedata_new

# %% Sync, align and normalize


def catch_lengths(data: dict):
    """catch error of unequal long sequenes of vicon and detectron data"""
    # compat for dec files
    datal1 = len(data['vic'])
    datal2 = len(data['dec'])
    datal3 = len(data['imu'])

    min_l = min([datal1, datal2, datal3])

    data['vic'] = data['vic'][:min_l, :, :]
    data['dec'] = data['dec'][:min_l, :, :]
    data['imu'] = data['imu'][:min_l, :, :]


def sync_data(model, data: dict, point=2, coo=1, verbose=False, start_steps=100, end_steps=50, max_steps=500):
    """sync data by looking at a spec point and coordinates(coo) and align"""
    catch_lengths(data)

    vic = model.get_knee_angles(data['vic'], case=0)[:, 0]  # only left angle here!
    imu = model.get_knee_angles(data['imu'], case=0)[:, 0]

    b, a = signal.butter(8, 0.05)
    vic = signal.filtfilt(b, a, vic)
    imu = signal.filtfilt(b, a, imu)
    vic = vic / np.max(vic)
    imu = imu / np.max(imu)

    if verbose:
        plt.figure('sync')
        plt.plot(vic)
        plt.plot(imu)

    vicmin = get_align_center(vic, verbose=verbose)
    imumin = get_align_center(imu, verbose=verbose)

    delta = imumin - vicmin

    print(f'DELTA: {delta}') if verbose else None

    if abs(delta) > 500:
        print(
            f"Warning!: Alignment too large: {delta} Steps. Setting delta to 0.")
        delta = 0

    if delta > 0:
        data['imu'] = data['imu'][delta:, :, :]
        center = vicmin
    else:
        data['vic'] = data['vic'][-delta:, :, :]
        data['dec'] = data['dec'][-delta:, :, :]
        center = imumin

    base_len = len(data['vic']) - end_steps
    len_traj = base_len - (center-start_steps)
    if len_traj > max_steps:
        base_len = (center-start_steps) + max_steps

    for key in data.keys():
        data[key] = data[key][(center-start_steps):base_len, :, :]


def get_align_center(x, act=0.2, str_r=[0.1, 0.2], verbose=False, mode=1, get_center=True):
    """
    synchronise by finding the first relevant peak, which is above the activation (act).
    the compare value is the mean value of the starting range (str_r)
    """
    # get starting mean value
    x = x*mode
    start_range = list(range(int(len(x)*str_r[0]), int(len(x)*str_r[1])))
    mean_val = np.mean(x[start_range])
    # define movement range
    ran = np.max(x) - np.min(x)
    # get all peaks in array
    peaks, _ = signal.find_peaks(x, height=0)

    # list of relevant peaks
    acc_peaks = []
    # remove to small peaks
    for peak in peaks:
        # decide
        abs_func = abs if get_center else lambda x: x

        if abs_func(x[peak] - mean_val) > (act*ran):
            acc_peaks.append(peak)

    # prevent err if no peak selected
    center = int(np.mean(acc_peaks[:1])) if len(acc_peaks) > 1 else 1e8

    if verbose:
        x = x*mode
        plt.plot(peaks, x[peaks], "x")
        plt.plot(acc_peaks, x[acc_peaks], "o")

    # decide wtheter to return sync or max peak points
    if get_center:
        return center
    else:
        return acc_peaks
# %% unify to same length


def unify(basemodel, data: dict):
    """
    goal is to apply the same length to all models in 
    oder to get similar lengths for all
    """
    for key in data.keys():
        data[key] = basemodel.rescale(data[key])


# %% read and sync


def read_all_data(model, path: str, cur_file: str, verbose=False):
    """
    read all datafiles for 1 case, align and sync
    """
    base = True if type(model) == BaseModel else False
    data = {
        'vic': read_vicdata(f'{path}/{cur_file}.csv', base=base),
        'dec': read_decdata(f'{path}/{cur_file}.json', base=base),
        'imu': read_imudata(f'{path}/{cur_file}_imu.json', base=base),
    }
    sync_data(model, data, verbose=verbose)
    unify(model, data)
    return data


# %% error analysis

def helper_mse(func, data, ax=None):
    fac = 180 / math.pi

    vic_angle = func(data['vic']) * fac
    dec_angle = func(data['dec']) * fac
    imu_angle = func(data['imu']) * fac

    mse_dec = mse_err(dec_angle, vic_angle, ax=ax)
    mse_imu = mse_err(imu_angle, vic_angle, ax=ax)

    std_dec = mse_std(dec_angle, vic_angle, ax=ax)
    std_imu = mse_std(imu_angle, vic_angle, ax=ax)

    return [mse_dec, std_dec], [mse_imu, std_imu]


def calc_mse_angle(basemodel, data: dict, ax=None):
    knee_dec, knee_imu = helper_mse(basemodel.get_knee_angles, data, ax=ax)

    knee_mse_dec = {
        'mse': knee_dec[0].copy(),
        'std': knee_dec[1].copy(),
    }

    knee_mse_imu = {
        'mse': knee_imu[0].copy(),
        'std': knee_imu[1].copy(),
    }

    if type(basemodel) is not BaseModel:
        hip_dec, hip_imu = helper_mse(basemodel.get_hip_angles, data, ax=ax)

        hip_mse_dec = {
            'mse': hip_dec[0],
            'std': hip_dec[1],
        }

        hip_mse_imu = {
            'mse': hip_imu[0],
            'std': hip_imu[1],
        }

        mse_dec = {
            'knee': knee_mse_dec,
            'hip': hip_mse_dec,
        }

        mse_imu = {
            'knee': knee_mse_imu,
            'hip': hip_mse_imu,
        }
    else:
        mse_dec = knee_mse_dec
        mse_imu = knee_mse_imu


    return mse_dec, mse_imu


def mse_err(a, b, ax=None, round_to=3):
    """mean squared error rounded to x digits"""
    return round(np.sqrt((np.square(a - b)).mean(axis=ax)), round_to)


def mse_std(a, b, ax=None, round_to=3):
    """mean squared error standard deviation rounded to x digits"""
    return round(np.sqrt((np.square(a - b)).std(axis=ax)), round_to)


def do_file_job(model, task_res, maxlist, task_path, file, movies=False, save=False):
    """
    complete all necessary preparation and calculation for one file
    """

    if file.endswith('.csv'):
        filename = file.split('.')[0]

        # --------------------------------------------------------
        data = read_all_data(model, task_path, filename)
        mse_dec, mse_imu = calc_mse_angle(model, data)

        task_res['dec'].append(mse_dec)

        if mse_imu['knee']['mse'] < 10 and mse_imu['hip']['mse'] < 10:
            task_res['imu'].append(mse_imu)

        # get max angles:
        max_angles = get_max_angles(data, model)
        maxlist.append(max_angles)

        print(f'\n{filename}:')

        if movies:
            make_movie(model, data,
                       f'../results/gif/{filename}.gif', title=filename)

        if save:
            plot_angle(model.get_knee_angles, 'knee', data, filename, lr=0)
            plt.savefig(f'../results/angles/{filename}_knee_l.png')

            plot_angle(model.get_knee_angles, 'knee', data, filename, lr=1)
            plt.savefig(f'../results/angles/{filename}_knee_r.png')

            if type(model) is not BaseModel:
                plot_angle(model.get_hip_angles, 'hip', data, filename, lr=0)
                plt.savefig(f'../results/angles/{filename}_hip_l.png')

                plot_angle(model.get_hip_angles, 'hip', data, filename, lr=1)
                plt.savefig(f'../results/angles/{filename}_hip_r.png')

            bland_altmann_analysis_for_data(data, model, filename)
        # --------------------------------------------------------


def get_all_errors(basemodel, movies=False, save=False):
    """
    calculate all mse errors of all files
    """
    basepath = '../data'

    result = {}
    max_angles = {}

    for pat in tqdm(os.listdir(basepath)):
        pat_data = {}
        max_pat = {}

        for task in os.listdir(f'{basepath}/{pat}'):
            task_path = f'{basepath}/{pat}/{task}'

            task_res = {
                'dec': [],
                'imu': [],
            }

            maxlist = []

            for file in os.listdir(task_path):
                do_file_job(basemodel, task_res, maxlist, task_path,
                            file, movies=movies, save=save)

            pat_data[task] = task_res
            max_pat[task] = maxlist

        result[pat] = pat_data
        max_angles[pat] = max_pat

    with open('../results/result.json', 'w') as f:
        json.dump(result, f,  indent=2)
    with open('../results/max_angles.json', 'w') as f:
        json.dump(max_angles, f,  indent=2)

    return result, max_angles

# %%


def bland_altmann_analysis_for_data(data, basemodel, filename):
    """Define Agreement of two Measurement Analyses"""

    fac = 180 / math.pi

    vic_angle = basemodel.get_knee_angles(data['vic']) * fac
    dec_angle = basemodel.get_knee_angles(data['imu']) * fac

    bland_altmann_analysis(
        vic_angle[:, 0], dec_angle[:, 0], 'Left_Knee', filename)
    bland_altmann_analysis(
        vic_angle[:, 1], dec_angle[:, 1], 'Right_Knee', filename)

    if type(basemodel) is not BaseModel:
        vic_angle = basemodel.get_hip_angles(data['vic']) * fac
        dec_angle = basemodel.get_hip_angles(data['imu']) * fac

        bland_altmann_analysis(
            vic_angle[:, 0], dec_angle[:, 0], 'Left_Hip', filename)
        bland_altmann_analysis(
            vic_angle[:, 1], dec_angle[:, 1], 'Right_Hip', filename)


def bland_altmann_analysis(baseline, testline, title, filename):
    """define the bland-altmann plot"""

    pair_y_axis = testline - baseline
    pair_x_axis = [(test + base) / 2 for test, base in zip(testline, baseline)]

    d = np.mean(pair_y_axis, axis=0)
    sd = np.std(pair_y_axis, axis=0)

    l1 = d + 2 * sd
    l2 = d - 2 * sd

    plt.figure(f'{filename} - {title}')
    plt.clf()
    plt.title(title)
    plt.grid(0.25)
    plt.plot(pair_x_axis, pair_y_axis, "^")
    xlimit = plt.xlim()
    plt.plot(xlimit, [d, d], 'b')
    plt.plot(xlimit, [l1, l1], '--')
    plt.plot(xlimit, [l2, l2], '--')
    plt.xlabel('Mean Measurement Angle (°)')
    plt.ylabel('Difference Between IMU and VICON (°)')

    plt.savefig(f'../results/bland_altmann/{filename}_{title}')


def get_peaks(data, func, leftright):
    """
    get the angles we require
    leftright: left = 0; right = 1
    """

    vic = func(data['vic'])[:, leftright]

    b, a = signal.butter(8, 0.05)
    vic = signal.filtfilt(b, a, vic)
    vic = vic / np.max(vic)

    peaks = get_align_center(vic, get_center=False, act=0.2)

    return peaks


def get_max_angles_for_leftright(data, model, leftright):
    """
    get the maxium values for each angle depedning on leftright = (0, 1)
    """
    rad2deg = 180 / math.pi
    # get peaks
    peaks = get_peaks(data, model.get_knee_angles, leftright=leftright)

    vic = model.get_knee_angles(data['vic'])[:, leftright] * rad2deg
    imu = model.get_knee_angles(data['imu'])[:, leftright] * rad2deg
    dec = model.get_knee_angles(data['dec'])[:, leftright] * rad2deg

    direction = 'left' if leftright == 0 else 'right'

    result_knee = {
        f'vic_max_{direction}': list(vic[peaks]),
        f'imu_max_{direction}': list(imu[peaks]),
        f'dec_max_{direction}': list(dec[peaks]),
    }

    if type(model) is not BaseModel:
        peaks = get_peaks(data, model.get_hip_angles, leftright=leftright)

        vic = model.get_hip_angles(data['vic'])[:, leftright] * rad2deg
        imu = model.get_hip_angles(data['imu'])[:, leftright] * rad2deg
        dec = model.get_hip_angles(data['dec'])[:, leftright] * rad2deg

        result_hip = {
            f'vic_max_{direction}': list(vic[peaks]),
            f'imu_max_{direction}': list(imu[peaks]),
            f'dec_max_{direction}': list(dec[peaks]),
        }

        result = {
            'knee': result_knee,
            'hip': result_hip,
        }

    else:
        result = result_knee

    return result


def get_max_angles(data, model):
    """
    extract the maximum angles from the data for left and right seprartely
    """
    result_l = get_max_angles_for_leftright(data, model, leftright=0)
    result_r = get_max_angles_for_leftright(data, model, leftright=1)

    final_result = {
        'left': result_l,
        'right': result_r,
    }

    return final_result

# %%

def run_analysis_by_name(file, movie=False):
    """run analysis of one file"""

    basemodel = BaseExtModel()
    subpath = 'Gang' if 'Gang' in file else 'Treppe'
    path = f'../data/{file[:2]}/{subpath}'

    data = read_all_data(basemodel, path, file, verbose=True)

    plot_angle(basemodel.get_knee_angles, 'knee', data, file, lr=0)
    plot_angle(basemodel.get_knee_angles, 'knee', data, file, lr=1)
    mse_dec, mse_imu = calc_mse_angle(basemodel, data)

    print(f'DEC-Err: {round(mse_dec[0], 2)}° ± {round(mse_dec[1], 2)}°')
    print(f'IMU-Err: {round(mse_imu[0], 2)}° ± {round(mse_imu[1], 2)}°')

    viz_all_models(0, data, basemodel, title=None, extend=False)
    bland_altmann_analysis_for_data(data, basemodel, file)

    if movie:
        make_movie(basemodel, data,  f'../results/gif/{file}.gif', title=file)


def load_res(file_path='../results/result.json'):
    with open(file_path) as f:
        data = json.load(f)
    return data

def eval_results(data, sel_mode='Gang', sel_sensor='dec', sel_joint='knee', val='mse'):
    """
    Extracts selected evaluation results from a JSON file.
    
    Args:
    - sel_mode (str): mode to filter on, default 'Gang'
    - sel_sensor (str): sensor to filter on, default 'dec'
    - sel_joint (str): joint to filter on, default 'knee'
    - val (str): evaluation metric to extract, default 'mse'
    
    Returns:
    - A list of evaluation results.
    """
    res = []
    for _, pat_data in data.items():
        if sel_mode in pat_data:
            sensor_data = pat_data[sel_mode].get(sel_sensor)
            if sensor_data is not None:
                loc_res = [loc[sel_joint][val] for loc in sensor_data]
                res.extend(loc_res)
    mean = np.mean(res)
    std = np.std(res)
    print(f'{sel_mode} {sel_sensor} {sel_joint}: {round(mean, 2)}° ± {round(std, 2)}°')

    return res

def analyse_res():
    data = load_res()
    for sel_sensor in ['dec', 'imu']:
        for sel_mode in ['Gang', 'Treppe']:
            for sel_joint in ['knee', 'hip']:
                eval_results(data, sel_joint=sel_joint, sel_sensor=sel_sensor, sel_mode=sel_mode)

# %% testing / running
if __name__ == '__main__':
    basemodel = BaseExtModel()
    result, max_angles = get_all_errors(basemodel, movies=False, save=False)
    analyse_res()

# %%



# %%
