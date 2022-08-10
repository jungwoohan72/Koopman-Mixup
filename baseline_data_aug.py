import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import random

import faiss

from d3rlpy.dataset import MDPDataset

def rad_data_aug(args, 
                dataset,
                n_aug = 0,
                random_seed=1):

    np.random.seed(random_seed)

    if args.scale: # RAD Scale
        if args.additional: # Offline RL style data addition
            observations = dataset.observations
            actions = dataset.actions
            rewards = dataset.rewards
            terminals = dataset.terminals

            idx = np.random.randint(0, len(dataset.observations)-1, size = n_aug)

            obs, acts, rews, epi_tems, tems = to_trans(dataset, idx, n_aug)

            if args.single_flag: # RAS-S
                if (obs.shape[0]) % 2 == 1:
                    obs = obs[:-1]

                random_number = np.random.uniform(0.5, 1.5, int(obs.shape[0]/2))
                random_number = np.expand_dims(random_number, 1)                    

                random_number = np.concatenate((random_number, random_number), axis=1)
                random_number = random_number.reshape(-1,1)
                
                obs = obs*random_number

                print("Single-Random number shape: {}".format(random_number.shape))
                print(random_number)

            else: # RAS-M
                if (obs.shape[0]) % 2 == 1:
                    obs = obs[:-1]

                random_number = np.random.uniform(0.5, 1.5, int(obs.shape[0]/2)*obs.shape[1]).reshape(int(obs.shape[0]/2), obs.shape[1])
                random_number = np.concatenate((random_number, random_number), axis=1)
                random_number = random_number.reshape(obs.shape[0], -1)

                print(random_number)
                print(obs)

                obs = obs*random_number

                print(obs)

                print("Multi-Random number shape: {}".format(random_number.shape))
                print(random_number)

            # Add to original dataset
            observations = np.concatenate((observations, obs), axis = 0)
            actions = np.concatenate((actions, acts), axis = 0)
            rewards = np.concatenate((rewards, rews))
            epi_terminals = np.concatenate((terminals, epi_tems))
            terminals = np.concatenate((terminals, tems))

            # Genearate MDPDataset
            new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

            n_epochs = int(1000000/((len(dataset.observations)+n_aug)/args.CQL_batch_size))

            print("RAD Additional data with uniform noise is used")
            print('MDP dataset generated')
            print(len(dataset.observations), n_aug, len(observations))
            return new_dataset, n_epochs

        else: # RAD scale over whole dataset without additional augmented data
            if args.single_flag:
                obs = dataset.observations
                random_number = np.random.uniform(0.5, 1.5, obs.shape[0]).reshape(-1,1)
                print("Single-Random number shape: {}".format(random_number.shape))
            else:
                obs = dataset.observations
                random_number = np.random.uniform(0.5, 1.5, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
                print("Multi-Random number shape: {}".format(random_number.shape))

            actions = dataset.actions
            rewards = dataset.rewards
            terminals = dataset.terminals

            observations = dataset.observations*random_number

            new_dataset = MDPDataset(observations, actions, rewards, terminals)

            n_epochs = int(1000000/((len(dataset.observations))/args.CQL_batch_size))

            print("RAD Only original data with uniform noise is used")

            return new_dataset, n_epochs

    else: # RAD Gaussian Noise Addition
        if args.additional: # Offline RL style augmented data addition
            observations = dataset.observations
            actions = dataset.actions
            rewards = dataset.rewards
            terminals = dataset.terminals

            idx = np.random.randint(0, len(dataset.observations)-1, size = n_aug)

            obs, acts, rews, epi_tems, tems = to_trans(dataset, idx, n_aug)

            if (obs.shape[0]) % 2 == 1:
                obs = obs[:-1]

            random_number = np.random.normal(0, 1, int(obs.shape[0]/2)*obs.shape[1]).reshape(int(obs.shape[0]/2), obs.shape[1])
            random_number = np.concatenate((random_number, random_number), axis=1)
            random_number = random_number.reshape(obs.shape[0], -1)

            obs = obs + random_number

            # Add to original dataset
            observations = np.concatenate((observations, obs), axis = 0)
            actions = np.concatenate((actions, acts), axis = 0)
            rewards = np.concatenate((rewards, rews))
            epi_terminals = np.concatenate((terminals, epi_tems))
            terminals = np.concatenate((terminals, tems))

            # Genearate MDPDataset
            new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

            n_epochs = int(1000000/((len(dataset.observations)+n_aug)/args.CQL_batch_size))

            print("RAD Additional data with gaussian noise is used")
            print('MDP dataset generated')
            print(random_number)

            print(len(dataset.observations), n_aug, len(observations))
            return new_dataset, n_epochs

        else: # RAD Gaussian noise addition to the whole dataset without additional augmented data
            obs = dataset.observations

            # Following RAD implementation -> masking half of the noise so that only the half of the data is augmented
            noise = np.random.normal(0, 1, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
            mask = np.random.uniform(0, 1, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0], -1) < 0.5
            noise = noise * mask

            actions = dataset.actions
            rewards = dataset.rewards
            terminals = dataset.terminals

            observations = dataset.observations + noise

            new_dataset = MDPDataset(observations, actions, rewards, terminals)

            n_epochs = int(1000000/((len(dataset.observations))/args.CQL_batch_size))

            print("RAD Only original data with gaussian noise is used")

            return new_dataset, n_epochs

def s4rl_data_aug(args, 
                dataset,
                n_aug = 0,
                random_seed=1):

    np.random.seed(random_seed)

    if args.additional: # Offline RL style augmented data addition
        observations = dataset.observations
        actions = dataset.actions
        rewards = dataset.rewards
        terminals = dataset.terminals

        idx = np.random.randint(0, len(dataset.observations)-1, size = n_aug)

        obs, acts, rews, epi_tems, tems = to_trans(dataset, idx, n_aug)

        if args.gaussian:
            # Gaussian noise addition
            noise = np.random.normal(0, 0.0003, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
            print("S4RL Additional data with gaussian noise is used")

        else:
            # Uniform noise addition
            noise = np.random.uniform(-0.0003, 0.0003, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
            print("S4RL Additional data with uniform noise is used")

        obs = obs + noise

        # Add to original dataset
        observations = np.concatenate((observations, obs), axis = 0)
        actions = np.concatenate((actions, acts), axis = 0)
        rewards = np.concatenate((rewards, rews))
        epi_terminals = np.concatenate((terminals, epi_tems))
        terminals = np.concatenate((terminals, tems))

        # Genearate MDPDataset
        new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

        n_epochs = int(1000000/((len(dataset.observations)+n_aug)/args.CQL_batch_size))

        print('MDP dataset generated')
        print(len(dataset.observations), n_aug, len(observations))
        return new_dataset, n_epochs

    else: # Augmentation on the whole dataset without addtional augmented data
        obs = dataset.observations

        if args.gaussian:
            # Gaussian noise addition
            noise = np.random.normal(0, 0.0003, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
            mask = np.random.uniform(0, 1, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0], -1) < 0.5
            noise = noise * mask
            print("S4RL Only original data with gaussian noise is used")
        else:
            # Uniform noise addition
            noise = np.random.uniform(-0.0003, 0.0003, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0],-1)
            mask = np.random.uniform(0, 1, obs.shape[0]*obs.shape[1]).reshape(obs.shape[0], -1) < 0.5
            noise = noise * mask
            print("S4RL Only original data with uniform noise is used")

        actions = dataset.actions
        rewards = dataset.rewards
        terminals = dataset.terminals

        observations = dataset.observations + noise

        new_dataset = MDPDataset(observations, actions, rewards, terminals)

        n_epochs = int(1000000/((len(dataset.observations))/args.CQL_batch_size))

        return new_dataset, n_epochs

def nmer_data_aug(args, 
                dataset,
                n_aug = 0,
                random_seed=1):

    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    terminals = dataset.terminals

    # Normalization
    cat = np.concatenate((observations, actions), axis=1)
    mean = np.mean(cat, axis=(0,1))
    std = np.std(cat, axis=(0,1))

    cat_norm = (cat - mean)/std

    aug_num = 0
    batch_size = n_aug

    # Selection of data to be augmented
    idx = np.random.randint(0, len(dataset.observations)-1, size = batch_size)

    # Find nearest neighbor of the selected data
    index = faiss.IndexFlatL2(cat.shape[1])
    index.add(cat_norm)

    k = 1 # 1st nearest neighbor
    D, I = index.search(cat_norm[idx], k)
    I = np.squeeze(I)

    obs_1 = observations[idx]
    acts_1 = actions[idx]
    next_obs_1 = observations[idx+1]
    rews_1 = rewards[idx][:, None]
    next_acts_1 = actions[idx+1]
    next_rews_1 = rewards[idx+1][:, None]

    obs_2 = observations[I]
    acts_2 = actions[I]
    next_obs_2 = observations[I+1]
    rews_2 = rewards[I][:, None]
    next_acts_2 = actions[I+1]
    next_rews_2 = rewards[I+1][:, None]

    # Perform mixup
    lam = np.random.beta(args.alpha, args.alpha , size = (batch_size,1))
    lam = np.float64(lam)
    
    obs_mix = lam * obs_1 + (1-lam)*obs_2
    acts_mix = lam * acts_1 + (1-lam)*acts_2
    obs_mix_n = lam * next_obs_1 + (1-lam)*next_obs_2
    rews_mix = lam * rews_1 + (1-lam)*rews_2
    acts_mix_n = lam * next_acts_1 + (1-lam)*next_acts_2
    rews_mix_n = lam * next_rews_1 + (1-lam)*next_rews_2

    rews_mix = np.squeeze(rews_mix)
    rews_mix_n = np.squeeze(rews_mix_n)
    
    obs = np.zeros((batch_size*2, obs_mix.shape[1])) # s, a, r, tem, epi_tem for one timestep. two data points will make one transition (s, a, r, s')
    acts = np.zeros((batch_size*2, acts_mix.shape[1]))
    rews = np.zeros(batch_size*2)
    tems = np.zeros(batch_size*2)
    epi_tems = np.zeros(batch_size*2)
    prev_list = np.linspace(0,batch_size*2-2,batch_size, dtype = int)
    next_list = prev_list +1

    obs[prev_list,:] = obs_mix
    obs[next_list,:] = obs_mix_n
    acts[prev_list,:] = acts_mix
    acts[next_list,:] = acts_mix_n
    rews[prev_list] = rews_mix
    rews[next_list] = rews_mix_n
    tems[prev_list] = 0
    tems[next_list] = 0 # Environment terminal is False
    epi_tems[prev_list] = 0
    epi_tems[next_list] = 1 # Episode terminal is True to make the episode has one transition.

    # Add to original dataset
    observations = np.concatenate((observations, obs), axis = 0)
    actions = np.concatenate((actions, acts), axis = 0)
    rewards = np.concatenate((rewards, rews))
    epi_terminals = np.concatenate((terminals, epi_tems))
    terminals = np.concatenate((terminals, tems))

    # Generate MDPDataset
    new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

    n_epochs = int(1000000/((len(dataset.observations)+n_aug)/args.CQL_batch_size))

    # episode.transitions
    print("Additional data with NMER augmentation is used")
    print('MDP dataset generated')
    print(len(dataset.observations), n_aug, len(observations))
    return new_dataset, n_epochs

def to_trans(dataset, idx, n_aug):
    
    obs = np.zeros((n_aug*2, dataset.observations.shape[1])) ## s,a,r,tem, epi_tem for one timestep. two data points will make one transition (s, a, r, s')
    acts = np.zeros((n_aug*2, dataset.actions.shape[1]))
    rews = np.zeros((n_aug*2))
    tems = np.zeros(n_aug*2)
    epi_tems = np.zeros(n_aug*2)
    prev_list = np.arange(0, len(obs)-1, 2)
    next_list = prev_list + 1

    obs[prev_list,:] = dataset.observations[idx,:]
    obs[next_list,:] = dataset.observations[idx+1,:]
    acts[prev_list,:] = dataset.actions[idx,:]
    acts[next_list,:] = dataset.actions[idx+1,:]
    rews[prev_list] = dataset.rewards[idx]
    rews[next_list] = dataset.rewards[idx+1]
    tems[prev_list] = 0
    tems[next_list] = 0 ## environment terminal is False
    epi_tems[prev_list] = 0
    epi_tems[next_list] = 1 ## episode terminal is True to make the episode has one transition.

    return obs, acts, rews, epi_tems, tems