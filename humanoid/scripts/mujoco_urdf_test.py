
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg
import torch
import argparse

parser = argparse.ArgumentParser(description='Deployment script.')
# parser.add_argument('--load_model', type=str, required=True,
#                     help='Run to load from.')
parser.add_argument('--terrain', action='store_true', help='terrain or plane')
args = parser.parse_args()

class Sim2simCfg(XBotLCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/quad.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def run_mujoco(cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    count_lowlevel =0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        # q, dq, quat, v, omega, gvec = get_obs(data)
        # q = q[-cfg.env.num_actions:]
        # dq = dq[-cfg.env.num_actions:]

        # # 1000hz -> 100hz
        # if count_lowlevel % cfg.sim_config.decimation == 0:

        #     obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
        #     eu_ang = quaternion_to_euler_array(quat)
        #     eu_ang[eu_ang > math.pi] -= 2 * math.pi

        #     obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
        #     obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)
        #     obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        #     obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
        #     obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        #     obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
        #     obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
        #     obs[0, 29:41] = action
        #     obs[0, 41:44] = omega
        #     obs[0, 44:47] = eu_ang

        #     obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

        #     hist_obs.append(obs)
        #     hist_obs.popleft()

        #     policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
        #     for i in range(cfg.env.frame_stack):
        #         policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            # action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            # target_q = action * cfg.control.action_scale


        # target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        # tau = pd_control(target_q, q, cfg.robot_config.kps,
        #                 target_dq, dq, cfg.robot_config.kds)  # Calc torques
        # tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        # data.ctrl = tau

        # mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

run_mujoco(Sim2simCfg())
