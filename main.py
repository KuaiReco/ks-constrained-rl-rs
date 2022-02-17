import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch
import logging
import random
import yaml
from types import SimpleNamespace as SN
import pprint
from algos import *
from utils import *
from buffer import *
from data_process import *
import time 
from torch.distributions.normal import Normal
import os

def is_debug_mode():  
    import os  
    return "DEBUG_MODE" in os.environ and os.environ["DEBUG_MODE"] == "true"

def get_path():
	import os
	return os.getcwd()

root_path = get_path()
# "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger


# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment()
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(root_path, "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    # run the framework
    run(_run, _config, _log)


def parse_config_file(params):
    config_file = "ddpg"
    for i, v in enumerate(params):
        print(v)
        if v.split("=")[0] == "--config":
            config_file = v.split("=")[1]
            del params[i]
            break
    return config_file

def Create_Policy(args):
    if args.algo=="ddpg":
        return DDPG.DDPG(args)
    if args.algo=="td3":
        return TD3.TD3(args)
    if args.algo=="td3_bc":
        return TD3_BC.TD3_BC(args)
    if args.algo=="a3c":
        return A3C.A3C(args)
    if args.algo=="a3c_loc":
        return A3C.A3C(args)
    if args.algo == "multi_critic_a3c":
        return MULTI_CRITIC_A3C.MULTI_CRITIC_A3C(args)
    if args.algo == "sac":
        return SAC.SAC(args)
    if args.algo=="deep_fm":
        return DEEP_FM.DEEP_FM(args)
    if args.algo=="wide_deep":
        return WIDE_DEEP.WIDE_DEEP(args)
    if args.algo=="awac":
        return AWAC.AWAC(args)
    if args.algo=="multi_critic_awac":
        return MULTI_CRITIC_AWAC.MULTI_CRITIC_AWAC(args)
    if args.algo=="awac_ddpg":
        return AWAC_DDPG.AWAC_DDPG(args)
    if args.algo=="multi_critic_awac_ddpg":
        return MULTI_CRITIC_AWAC_DDPG.MULTI_CRITIC_AWAC_DDPG(args)
    
def offline_ab(policy, behavior_policy, replay_buffer):
    # calculate sequentially, could be calculate in batch as well.
    total_ratio=0
    estimated_rewards = np.zeros(8)
    with torch.no_grad():
        state = replay_buffer.state
        action = replay_buffer.action
        reward = replay_buffer.response
        size = replay_buffer.size
        start_time = time.time()
        #print(f"We have {len(state)} samples in total")
        for t, s in enumerate(state):
            s = np.array(s)
            policy_action = policy.select_action(s)
            behavior_policy_action = behavior_policy.select_action(s)

            dist = Normal(policy_action, 25)
            dist_b = Normal(behavior_policy_action, 25)

            a = torch.tensor(action[t].reshape(1, -1)).to(device)

            log_prob = dist.log_prob(a).sum(axis=-1)
            log_prob_b = dist_b.log_prob(a).sum(axis=-1)
            ratio = torch.exp(log_prob - log_prob_b).clamp(0,10)
            total_ratio = total_ratio + ratio

        print(f"Finished calculating total_ratio {total_ratio}. Time elapsed {time.time() - start_time}")
        
        total_ratio = total_ratio
        for t, s in enumerate(state):
            s = np.array(s)
            s = np.reshape(s, [1, -1])
            policy_action=policy.select_action(s)
            behavior_policy_action=behavior_policy.select_action(s)
            dist=Normal(policy_action,25)
            dist_b=Normal(behavior_policy_action, 25)

            a=torch.tensor(action[t]).to(device)
            log_prob=dist.log_prob(a).sum(axis=-1)
            log_prob_b=dist_b.log_prob(a).sum(axis=-1)
            ratio=torch.exp(log_prob-log_prob_b).clamp(0,10)
            ratio = ratio/total_ratio

            R = np.array(reward[t])
            ratio = ratio.item()
            R = ratio * R
            

            estimated_rewards = estimated_rewards + R

        print(f"Time elapsed {time.time()-start_time}")
        print("estimated_rewards:", estimated_rewards)

    return estimated_rewards

def run(_run, _config, _log):
    args = SN(**_config)
    args.ex_results_path = os.path.join(args.ex_results_path, str(_run._id))
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                    indent=4,
                                    width=1)
    _log.info("\n\n" + experiment_params + "\n")

    if args.use_tensorboard:
        logger.setup_tb(args.ex_results_path)

    # sacred is on by default
    logger.setup_sacred(_run)

    start_time = time.time()
    logger.console_logger.info("Beginning training for {} steps".format(args.max_steps))
    
    last_train_e=0
    last_test_e=-args.test_every-1
    last_save_e=0
    last_log_e=0

    replay_buffer=ReplayBuffer(args)
    #loading from the training buffer

    replay_buffer.load(get_path() + "/data/tripadvisor_data/train_buffer_fulldata.npz")
    policy = Create_Policy(args)

    test_replay_buffer=ReplayBuffer(args)
    test_replay_buffer.load(get_path() + "/data/tripadvisor_data/test_buffer_fulldata.npz") #load test buffer
    behavior_policy=SL.SL(args)
    #load behavior_policy
    behavior_policy.load(get_path() + "/behavior_model/sl")

    sum_reward = 2
    best_rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    for e in range(int(args.max_steps)):
        loss = policy.train(replay_buffer, args, args.batch_size)
        #logger.log_stat("predict_action", loss[2].item(), e)
        # print(loss[0])
        if (e-last_test_e) / args.test_every >= 1.0: #testing
            pass
        if args.save_model and e % args.save_every == 0: #saving
            save_path = os.path.join(args.ex_results_path, "models/")
            os.makedirs(save_path, exist_ok=True)
            print(f'at ieration {e}, time elapsed {time.time() - start_time}, actor loss : {loss[0].item()}, critic loss : {loss[1].item()}')
            #print(loss)
            estimated_rewards = offline_ab(policy, behavior_policy, test_replay_buffer)
            if sum_reward < estimated_rewards[6] + estimated_rewards[7]:
                sum_reward = estimated_rewards[6] + estimated_rewards[7]
                best_rewards = estimated_rewards
                policy.save(save_path)
            
            print("best_rewards:", best_rewards)
            logger.log_stat(f"best_rewards: ", best_rewards[0], e)
            logger.log_stat("best_rewards: ", best_rewards[1], e)
            logger.log_stat("best_rewards: ", best_rewards[2], e)
            logger.log_stat("best_rewards: ", best_rewards[3], e)
            logger.log_stat("best_rewards: ", best_rewards[4], e)
            logger.log_stat("best_rewards: ", best_rewards[5], e)
            logger.log_stat("best_rewards: ", best_rewards[6], e)
            logger.log_stat("best_rewards: ", best_rewards[7], e)
        if e % args.log_every == 0:
            print(loss)
            #logger.log_stat("actor_loss", loss[0].item(), e)
            #logger.log_stat("critic_loss", loss[1].item(), e)
            

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    config_file = parse_config_file(params)

    ex.add_config(get_path() + '/config/{}.yaml'.format(config_file))

    logger.info(
        f"Saving to FileStorageObserver in {root_path}/results/{config_file}.")
    file_obs_path = os.path.join(results_path, config_file)
    ex.add_config(name=config_file)
    ex.add_config(ex_results_path=file_obs_path)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
 
