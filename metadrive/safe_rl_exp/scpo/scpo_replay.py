import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
import numpy as np
import torch
from torch.optim import Adam
import time
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
 
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.envs.gym_wrapper import createGymWrapper # import the wrapper
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

import gymnasium as gym
from metadrive.envs.gym_wrapper import createGymWrapper # import the wrapper

import os.path as osp
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def create_env(map_type):
    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: "X",  # 3 block
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }
    map_config["config"]= map_type

    lidar=dict(
        num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=True
    )
    vehicle_config = dict(lidar=lidar)

    # env = SafeMetaDriveEnv(dict(map_config = map_config))
    env = createGymWrapper(SafeMetaDriveEnv)(
        config={
            "use_render": True,
            "map_config": map_config,
            "vehicle_config": vehicle_config,
            "num_scenarios": 1,
            "accident_prob": 0.8,
            "start_seed": 100,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "out_of_route_done": False,
            "cost_to_reward": False,
            "crash_vehicle_penalty": 0.0,
            "crash_object_penalty": 0.0,
            "traffic_density": 0.55,
        }
    )  # wrap the environment
    return env


def replay(env_fn, model_path=None, max_epoch=1):
    if not model_path:
        print("please specify a model path")
        raise NotImplementedError
    # if not video_name:
    #     print("please specify a video name")
    #     raise NotImplementedError    
    
    # Instantiate environment
    env = env_fn()
    
    # reset environment
    o = env.reset()
    d = False
    ep_ret = 0
    time_step = 0
    epoch = 0
    M = 0. # initialize the current maximum cost
    o_aug = np.append(o, M) # augmented observation = observation + M 
    first_step = True
    
    # video_array = []
    
    # load the model 
    ac = torch.load(model_path)
    
    # evaluate the model 
    while True:
        time_step += 1
        if d:
            epoch += 1
            print('Episode Return: %.3f'%(ep_ret))
            if epoch == max_epoch:
                env.close()
                break
            ep_ret = 0
            o = env.reset()
            M = 0. # initialize the current maximum cost 
            # o_aug = o.append(M) # augmented observation = observation + M 
            o_bug = np.append(o, M) # augmented observation = observation + M 
            first_step = True
        
        try:
            a, v, vc, logp, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
        except:
            print('please choose the correct environment, the observation space doesn''t match')
            raise NotImplementedError
        

        try: 
            next_o, r, d, i = env.step(a)
            info = dict()
            info['cost_out_of_road'] = i.get('cost_out_of_road', 0)
            info['crash_vehicle_cost'] = i.get('crash_vehicle_cost', 0)
            info['crash_object_cost'] = i.get('crash_object_cost', 0)
            info['cost'] = info['cost_out_of_road'] + info['crash_vehicle_cost'] + info['crash_object_cost']

            assert 'cost' in info.keys()
            # linear_v = math.hypot(next_o[3],next_o[4])
            # linear_a = math.hypot(next_o[0],next_o[1])
            # print("speed: ", math.hypot(next_o[3],next_o[4]), "\tVelocity of the robot: ", (next_o[3],next_o[4],next_o[5]))
            # print("accel mag: ",math.hypot(next_o[0],next_o[1]),"\tAcceleration of the robot: ",(next_o[0],next_o[1],next_o[2]))

            # max_accel = 5
            # cost_accel = (linear_a - max_accel) / linear_a if linear_a > max_accel else 0

            # #Increment the cost in info
            # info['cost_accel'] = cost_accel
            # info['cost'] += cost_accel
        except: 
            # simulation exception discovered, discard this episode 
            next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
            info['cost'] = 0 # no cost when episode done    
        
        
        if first_step:
            # the first step of each episode 
            cost_increase = info['cost']
            M_next = info['cost']
            first_step = False
        else:
            # the second and forward step of each episode
            cost_increase = max(info['cost'] - M, 0)
            M_next = M + cost_increase 
        
        # Update obs (critical!)
        # o = next_o
        o_aug = np.append(next_o, M_next)

        # img_array = env.render(mode='rgb_array')
        # video_array.append(img_array)

        ep_ret += r


    # # save video 
    # fps = 60
    # dsize = (1920,1080)
    # out_path = '../video'
    # existence = os.path.exists(out_path)
    # if not existence:
    #     os.makedirs(out_path)
    # video_writer = cv2.VideoWriter(
    #         os.path.join(out_path, f'{video_name}.mp4'),
    #         cv2.VideoWriter_fourcc(*'mp4v'),  # Change 'FMP4' to 'mp4v'
    #         fps,
    #         dsize
    #     )

    # for frame in video_array:
    #     resized = cv2.resize(frame, dsize=dsize)
    #     video_writer.write(resized)

    # video_writer.release()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--max_epoch', type=int, default=1)  # the maximum number of epochs
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--map_type', type=str, default='SXCOY')

    args = parser.parse_args()

    replay(lambda : create_env(args.map_type), model_path=args.model_path, max_epoch=args.max_epoch)