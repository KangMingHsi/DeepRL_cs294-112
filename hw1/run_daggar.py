
# coding: utf-8

# In[1]:

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from sklearn.utils import shuffle
from model import BCModel, load_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('save_model', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    
    expert_data = load_data(args.expert_data)
    obs_data = np.array(expert_data['observations'])
    a_data = np.array(expert_data['actions'])
    
    batch_size = 16
    env = gym.make(args.envname)
    
    net_param = dict()
    net_param['d1'] = 128
    net_param['d2'] = 64
    net_param['d3'] = 32
    
    bc = BCModel(net_param=net_param, batch_size=batch_size, input_size=env.observation_space.shape[0],                  action_size=env.action_space.shape[0], epoch=20)
    
    for i in range(5):
        print('-------' + str(i) + '--------')
        n = obs_data.shape[0]
        obs_data, a_data = shuffle(obs_data, a_data, random_state = 0)

        train_num = int(0.7 * n)
        x_train = np.reshape(obs_data[:train_num], newshape=[-1, env.observation_space.shape[0]])
        y_train = np.reshape(a_data[:train_num], newshape=[-1, env.action_space.shape[0]])
        x_test = np.reshape(obs_data[train_num:], newshape=[-1, env.observation_space.shape[0]])
        y_test = np.reshape(a_data[train_num:], newshape=[-1, env.action_space.shape[0]])
    
    
        with tf.Session() as sess:

            tf_util.initialize()
            
            if i > 0:
                bc.load(args.envname+'_dag', sess)
            bc.fit(x_train, y_train, sess)
            bc.save(args.envname+'_dag', sess)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
         
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    bc_action = bc.predict([obs], sess)
                    ex_action = policy_fn(obs[None,:])

                    observations.append(obs)
                    actions.append(ex_action)
                    obs, r, done, _ = env.step(bc_action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            
            obs_data = np.concatenate((obs_data, observations))
            a_data = np.concatenate((a_data, actions))
        
    env.close()


if __name__ == '__main__':
    main()

