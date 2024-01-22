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


def create_env():
    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: "X",  # 3 block
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }
    map_config["config"] = "X"

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

def replay(env_fn, model_path=None, max_epoch=1, num_constraints=1):
    if not model_path:
        print("please specify a model path")
        raise NotImplementedError
        
    # Instantiate environment
    env = env_fn()
    
    # reset environment
    o = env.reset()
    d = False
    ep_ret = 0
    time_step = 0
    epoch = 0
    M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints

    o_aug = np.append(o, M) # augmented observation = observation + M 
    first_step = True
        
    # load the model 
    ac = torch.load(model_path)
    get_costs = lambda info, constraints: np.array([info[key] for key in constraints if key != 'cost'])
    constraints_list = []

    
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
            M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints
            # o_aug = o.append(M) # augmented observation = observation + M 
            o_aug = np.append(o, M) # augmented observation = observation + M 
            first_step = True
            break
        
        try:
            a, v, vc, logp, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
        except:
            print('please choose the correct environment, the observation space doesn''t match')
            raise NotImplementedError
        

        try: 
            next_o, r, d, i = env.step(a)
            info = dict()
            info['cost_out_of_road'] = i.get('cost_out_of_road', 0)
            # info['crash_vehicle_cost'] = i.get('crash_vehicle_cost', 0)
            # info['crash_object_cost'] = i.get('crash_object_cost', 0)
            info['cost'] = info['cost_out_of_road']# + info['crash_vehicle_cost'] + info['crash_object_cost']
            # info['cost_env'] =  i.get('cost', 0)
            # # info['cost_crash_vehicle'] = i.get('crash_vehicle_cost', 0)
            # # info['crash_object_cost'] = i.get('crash_object_cost', 0)
            # info['cost_acceleration']  = 0.5 if abs(i['acceleration']) > 0.4 else 0
            # info['cost_steer']  = 0.25 if abs(i['steering']) > 0.2 else 0
            # # random_number = random.randint(1, 1000)

            # # if random_number <= 57:
            # #     info['crash_vehicle_cost'] = 0.5
            # # elif random_number <= 91:
            # #     info['crash_object_cost'] =  0.32
            # # elif random_number <= 121:
            # #      info['cost_out_of_road'] = 0.69
            # info['cost'] = info['cost_env'] + info['cost_acceleration'] + info['cost_steer']
            assert 'cost' in info.keys()
        except: 
            # simulation exception discovered, discard this episode 
            next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
            info['cost'] = 0 # no cost when episode done    
        
        if first_step:
            # the first step of each episode 
            constraints_list = [key for key in info.keys()]
            cost_increase =  get_costs(info, constraints_list)
            # cost_increase = info['cost'] # define the new observation and cost for Maximum Markov Decision Process
            M_next = cost_increase
            first_step = False
        else:
            # the second and forward step of each episode
            costs = get_costs(info,constraints_list)
            cost_increase = np.maximum(costs - M, 0)
            M_next = M + cost_increase
             
        
        # Update obs (critical!)
        # o = next_o
        o_aug = np.append(next_o, M_next)

        ep_ret += r


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--max_epoch', type=int, default=1)  # the maximum number of epochs
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_constraints', type=int, default=1) # Number of constraints
    args = parser.parse_args()

    replay(lambda : create_env(), model_path=args.model_path, max_epoch=args.max_epoch, num_constraints=args.num_constraints)