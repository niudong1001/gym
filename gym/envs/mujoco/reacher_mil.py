import numpy as np
from scipy.spatial.distance import cdist
from gym import utils
from PIL import Image
import gc
import glob
import os
from natsort import natsorted
from gym.envs.mujoco import mujoco_env

class ReacherMILEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        gc.enable() # automatic garbage collection
        utils.EzPickle.__init__(self)
        self.xml_paths = natsorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/sim_vision_reach_test_xmls/*")))
        self.xml_iter = iter(self.xml_paths)
        self.n_distractors = 2 #2
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.next(), 5)
        # mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 5)
        self.eept_vel = np.zeros_like(self.get_body_com("fingertip"))

    def _step(self, a):
        prev_eept = self.get_body_com("fingertip")
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist# + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        # self.model.forward()
        curr_eept = self.get_body_com("fingertip")
        self.eept_vel = (curr_eept - prev_eept) / self.dt
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.
        self.viewer.cam.lookat[1] = 0.
        self.viewer.cam.lookat[2] = 0.
        self.viewer.cam.distance = 1.0#1.3
        self.viewer.cam.elevation = -90 #-90

    def reset_model(self):
        # qpos = self.init_qpos
        # qvel = self.init_qvel
        # cube_pos = np.random.rand(self.n_distractors+1, 2)
        # pair_dist = cdist(cube_pos, cube_pos)
        # pair_dist = pair_dist[pair_dist != 0]
        # while np.any(pair_dist < 0.15):
        #     cube_pos = np.random.rand(self.n_distractors+1, 2)
        #     pair_dist = cdist(cube_pos, cube_pos)
        #     pair_dist = pair_dist[pair_dist != 0]
        # body_offset = np.zeros((self.n_distractors+1, 2))
        # body_offset[0] = np.array([0.4*cube_pos[0, 0]-0.3, 0.4*cube_pos[0, 1]-0.1])
        # body_offset[1:, 0] = 0.4*cube_pos[1:, 0]-0.3
        # body_offset[1:, 1] = 0.4*cube_pos[1:, 1]-0.1
        # qpos[2:4] += body_offset[0]
        # qpos[9:11] += body_offset[1]
        # qpos[16:18] += body_offset[2]
        # # qvel[2:] = 0.
        # self.set_state(qpos, qvel)
        return self._get_obs()
    
    def next(self):
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.next(), 5)
        self.eept_vel = np.zeros_like(self.get_body_com("fingertip"))

    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((80,64), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([
            self.model.data.qpos.flat[:2],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip"),
            self.eept_vel
            ])

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:2],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip"),
            self.eept_vel
        ])
