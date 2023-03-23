# %%
import numpy as np
from numpy import sin, cos
import math

# Constant key-dict for reading vicon csv
VIC_KEYS = {
    'LASI': 0,
    'RASI': 1,
    'LPSI': 2,
    'RPSI': 3,
    'LTHI': 4,
    'LKNE': 5,
    'LTIB': 6,
    'LANK': 7,
    'LHEE': 8,
    'LTOE': 9,
    'RTHI': 10,
    'RKNE': 11,
    'RTIB': 12,
    'RANK': 13,
    'RHEE': 14,
    'RTOE': 15,
}

# angles calc inspired by:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249


def unit_vector(vector):
    """ Returns the unit vector of the vector"""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class BaseModel(object):
    """
    Base Model all three models will be transformed to, see model below:
    Syntax is: Number[Base]

       0[-1]
      /     |
    4[0]   1[0]
      |      |
    5[4]   2[1]
      |      |
    6[5]   3[2]

    """

    def __init__(self):
        self._parents = [-1, 0, 1, 2, 0, 4, 5]
        self._children = [[1, 4], [2], [3], [], [5], [6], []]
        self._right = [4, 5, 6]
        self._left = [1, 2, 3]
        self._center = [0]
        self._n_joints = 7
        # connections between joints
        self._connect = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]
        ]
        self._baselen = [
            0.5, 2, 2, 0.5, 2, 2
        ]
        self._legs = [
            [1, 2, 3],  # left
            [4, 5, 6],  # right
        ]

    def joints_right(self):
        return self._right

    def joints_left(self):
        return self._left

    def connect(self):
        return self._connect

    def parents(self):
        return self._parents

    def legs(self):
        return self._legs

    def calc_flex(self, upper, lower):
        """
        get the hip angle between 
        """
        dx1, len1 = upper[1], np.linalg.norm(upper)
        dx2, len2 = lower[1],  np.linalg.norm(lower)

        alpha = math.asin(dx1 / len1)
        beta = math.asin(dx2 / len2)

        return alpha - beta

    def get_knee_angles(self, basedata: np.ndarray, case=1):
        """
        return knee angles [timesteps, {right[0], left[1]}]
        """
        # init
        datalen = len(basedata)
        knee_angles = np.zeros([datalen, 2])

        # for all timesteps
        for timestep, loc_dat in enumerate(basedata):
            # get right(0), left (pos) and the leg connections
            for rightleft, leg in enumerate(self._legs):
                # get both vectors
                upper_leg = loc_dat[leg[1], :] - loc_dat[leg[0], :]
                lower_leg = loc_dat[leg[2], :] - loc_dat[leg[1], :]
                # calc angle
                if case:
                    knee_angles[timestep, rightleft] = self.calc_flex(
                        lower_leg, upper_leg)

                    if timestep > 0:
                        knee_angles[timestep, rightleft] -= knee_angles[0, rightleft]
                else:
                    knee_angles[timestep, rightleft] = angle_between(
                        upper_leg, lower_leg)
        if case:      
            knee_angles[0, :] = [0., 0.]

        return knee_angles

    def rescale(self, basedata: np.ndarray):
        """
        rescale the models to the uniform lengths
        basedata[timestep, points, xyz]
        """
        new_basedata = np.copy(basedata)

        # for all timesteps
        for timestep, loc_dat in enumerate(basedata):
            # for all points
            for (link, new_l) in zip(self._connect, self._baselen):
                # get rescaled vec
                vec = unit_vector(
                    loc_dat[link[1], :] - loc_dat[link[0], :]) * new_l
                # reapplied
                new_basedata[timestep, link[1],
                             :] = new_basedata[timestep, link[0], :] + vec
        return new_basedata


class BaseExtModel(BaseModel):
    """
    Base Model all three models will be transformed to, see model below:
    Syntax is: Number[Base]

       7[0]
        |
        |
       0[-1]
      /     |
    4[0]   1[0]
      |      |
    5[4]   2[1]
      |      |
    6[5]   3[2]

    """

    def __init__(self):
        self._parents = [-1, 0, 1, 2, 0, 4, 5, 0]
        self._children = [[1, 4, 7], [2], [3], [], [5], [6], [], []]
        self._right = [4, 5, 6]
        self._left = [1, 2, 3]
        self._center = [0]
        self._n_joints = 8
        # connections between joints
        self._connect = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7]
        ]
        self._baselen = [
            0.5, 2, 2, 0.5, 2, 2, 2
        ]
        self._legs = [
            [1, 2, 3],  # left
            [4, 5, 6],  # right
        ]

    def get_hip_angles(self, basedata: np.ndarray):
        """
        return knee angles [timesteps, {right[0], left[1]}]
        """
        # init
        datalen = len(basedata)
        hip_angles = np.zeros([datalen, 2])

        # for all timesteps
        for timestep, loc_dat in enumerate(basedata):
            # get the upward pointing hip vector
            hip_up = loc_dat[0, :] - loc_dat[7, :]

            # get right(0), left (pos) and the leg connections
            for rightleft, leg in enumerate(self._legs):
                # get both vectors
                upper_leg = loc_dat[leg[1], :] - loc_dat[leg[0], :]
                # calc angle
                hip_angles[timestep, rightleft] = self.calc_flex(
                    hip_up, upper_leg)
                if timestep > 0:
                    hip_angles[timestep, rightleft] -= hip_angles[0, rightleft]
        
        hip_angles[0, :] = [0., 0.]

        return hip_angles


class DecModel(BaseExtModel):
    """
    Extension to BaseExtModel from VideoPose3D.
    So it inherits from Base AND BaseExt Model.
    Syntax is: Number[Base]

         10[9]
          /
        9[8]
          |
         8[7]
          |
        _7[0]_
      /    |  |
    11[7]  |  14[7]
     |     |  |
    12[11] |  15[14]
    |      |  |
    13[12] |  16[15]
           |
        0[-1]
      /     |
    4[0]   1[0]
      |      |
    5[4]   2[1]
      |      |
    6[5]   3[2]
    """

    def __init__(self):
        super(DecModel, self).__init__()
        self._parents = [
            -1, 0, 1, 2, 0, 4, 5,  # base
            0, 7, 8, 9, 7, 11, 12, 7, 14, 15,  # extension
        ]
        self._children = [
            [1, 4, 7], [2], [3], [], [5], [6], [],  # basemodel
            [11, 8, 14], [9], [10], [],  # center upper
            [12], [13], [], [15], [16], [],  # left, right upper
        ]
        self._left = [4, 5, 6, 11, 12, 13]
        self._right = [1, 2, 3, 14, 15, 16]
        self._center = [0, 7, 8, 9, 10]
        self._n_joints = 17
        # connections between joints
        self._connect = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],  # basemodel
            [0, 7], [7, 8], [8, 9], [9, 10],  # center
            [7, 14], [14, 15], [15, 16],  # left
            [7, 11], [11, 12], [12, 13],  # right
        ]
        self._baselen = [
            0.5, 2, 2, 0.5, 2, 2,  # basemodel
            4, 1, 0.5, 0.5,  # center
            0.5, 2, 2,  # left
            0.5, 2, 2,  # right
        ]


class VicModel(BaseModel):
    """
    Extension to BaseModel from Vicon(Nexus).
    Syntax is: Name(Number), Joint: {}

        RPSI(3)---SACR(16)---LPSI(2)
        //                    //
    RASI(1)---------------LASI(0)
        |                    |
        |RTHI(10)            |LTHI (4)
        |                    |
    {RKNE(11)}           {LKNE (5)}
        |                    |
        |RTIB(12)            |LTIB (6)
        |                    |
    {RANK(13)}-RHEE(14)  {LANK(7)}-LHEE(8)
    //                   //
    RTOE(15)             LTOE(9)
    """

    def __init__(self):
        super(VicModel, self).__init__()
        self._parents = [
            16, 16, 16, 16,  # base
            0, 0, 5, 5,  # left leg
            7, 7,  # left foot
            1, 1, 11, 11,  # right leg
            13, 13,  # right foot
            -1,  # center
        ]
        self._left = [0, 2, 4, 5, 6, 7, 8, 9]
        self._right = [1, 3, 10, 11, 12, 13, 14, 15]
        self._center = [16]
        self._n_joints = 17
        self._connect = [
            [0, 1], [1, 3], [3, 2], [2, 0],  # base
            [2, 5], [5, 0],  # left upper leg
            [5, 7],  # left lower leg
            [7, 8], [7, 9], [8, 9],  # left foot
            [1, 11], [11, 3],  # right upper leg
            [11, 13],  # right lower leg
            [13, 14], [13, 15], [14, 15]  # right foot
        ]
        self._baselen = [
            1, 0.5, 0.5, 0.5,  # base
            2, 2,  # left upper leg
            2,  # left lower leg
            0.2, 0.2, 0.2,  # left foot
            2, 2,  # right upper leg
            2,  # right lower leg
            0.2, 0.2, 0.2  # right foot
        ]
        self._legs = [
            [0, 5, 7],  # left
            [1, 11, 13],  # right
        ]


class ViconCalc(object):
    """
    Provide the angle calc from the official VICON Documentation
    See docs/Plug-in Gait Reference Guide page 50.
    """

    def __init__(self, data, vicmodel: VicModel):
        # first get the interasis distance
        self.get_interasis_dist(data)
        # get left, right leg len and mean
        self.get_leg_lens(data, vicmodel)
        # next the following distances are needed:
        self.c = self.leglen_m * 0.115 - 15.3
        # marker radius in [mm]
        self.marker_r = 5
        # get the hip joint center offsets
        self.off_hjc_l = self.calc_offset_hip_joint_center(left=True)
        self.off_hjc_r = self.calc_offset_hip_joint_center(left=False)

    def get_interasis_dist(self, data):
        """mean distance of lasi and rasi"""
        lasi_all = data[:, VIC_KEYS['LASI'], :]
        rasi_all = data[:, VIC_KEYS['RASI'], :]

        # get all distances in each frame
        interasis_all = [np.linalg.norm(lasi - rasi)
                         for lasi, rasi in zip(lasi_all, rasi_all)]

        self.interasis = np.mean(interasis_all, axis=0)

    def get_leg_lens(self, data, vicmodel: VicModel):
        """Left Leg Length, Right Leg Length, Mean Leg Length"""
        leg_l, leg_r = vicmodel.legs()[0], vicmodel.legs()[1]
        self.leglen_l = self.calc_leg_len(leg_l, data)
        self.leglen_r = self.calc_leg_len(leg_r, data)
        self.leglen_m = 0.5 * (self.leglen_l + self.leglen_r)

    def calc_leg_len(self, leg, data):
        """
        calc leg len from hip-kne-ank acc to:
        https://docs.vicon.com/display/Nexus25/Take+subject+measurements+for+Plug-in+Gait
        """
        hip_all = data[:, leg[0], :]
        kne_all = data[:, leg[1], :]
        ank_all = data[:, leg[2], :]

        leglens = []

        for hip, kne, ank in zip(hip_all, kne_all, ank_all):
            upper = np.linalg.norm(hip - kne)
            lower = np.linalg.norm(kne - ank)
            leglens.append(upper + lower)

        leglen = np.mean(leglens, axis=0)

        return leglen

    def calc_asis_troc_dist(self, leglen):
        """page50 (I)"""
        return 0.1288 * leglen - 48.56

    def calc_offset_hip_joint_center(self, theta=0.5, beta=0.314, left=True):
        """this is the offset vector of the hip joint center - page 50"""
        leglen = self.leglen_l if left else self.leglen_r
        asis_troc_dist = self.calc_asis_troc_dist(leglen)

        x = self.c * cos(theta)*sin(beta) - \
            (asis_troc_dist + self.marker_r) * cos(beta)

        y = -(self.c * sin(theta) - self.interasis)

        z = -self.c * cos(theta) * cos(beta) - \
            (asis_troc_dist + self.marker_r) * sin(beta)

        # negate the y offset for the right leg:
        # see p.51 first line
        y = y if left else -y

        return np.array([x, y, z])

    def get_pelvis_coo(self, data):
        """define the pelvis coo system -> docs p.52"""

        # get all relevant data points:

        lasi_all = data[:, VIC_KEYS['LASI'], :]
        rasi_all = data[:, VIC_KEYS['RASI'], :]

        lpsi_all = data[:, VIC_KEYS['LPSI'], :]
        rpsi_all = data[:, VIC_KEYS['RPSI'], :]

        """The origin is taken as the midpoint of the two asis markers."""

        self.center_all = (lasi_all + rasi_all) / 2

        """
        Y axis, is the direction from the right asis marker to the left asis marker
        """

        self.y_all = [(lasi - rasi) / np.linalg.norm(lasi-rasi)
                      for lasi, rasi in zip(lasi_all, rasi_all)]

        """
        The secondary direction is taken as the direction 
        from the sacrum marker to the right asis marker
        """

        sacrum_all = (lpsi_all + rpsi_all) / 2
        sec_dir = [rpsi - sacrum for rpsi, sacrum in zip(rpsi_all, sacrum_all)]

        """
        The Z direction is generally
        upwards, perpendicular to this plane
        """

        self.z_all = [np.cross(y, sec) for y, sec in zip(self.y_all, sec_dir)]

        """
        the X axis is generally forwards
        """

        self.x_all = [np.cross(y, z) for y, z in zip(self.y_all, self.z_all)]

    # %%
