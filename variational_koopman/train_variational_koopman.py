import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import numpy as np
import progressbar
import random
import tensorflow as tf

import time

import wandb

from variational_koopman.replay_memory import ReplayMemory
from variational_koopman.variational_koopman_model import VariationalKoopman
# from utils import visualize_predictions, perform_rollouts

def training(args, x, u, state_dim, act_dim, dvk_model_train = False, config = {}, logging = False, dvk_model_dir="", random_seed=1):
    # Set random seed
    np.random.seed(random_seed)
    # Test
    # Create environment
    # env = gym.make(args.domain_name)

    # Find state and action dimensionality from environment
    args.state_dim = state_dim
    # if args.domain_name == 'CartPole-v1': args.state_dim += 1 # taking sine and cosine of theta
    args.action_dim = act_dim
    # args.action_max = env.action_space.high[0]

    # Construct model
    net = VariationalKoopman(args)

    # For logging
    config['num_epochs'] = args.num_epochs
    config['learning_rate'] = args.learning_rate
    config['decay_rate'] = args.decay_rate
    config['l2_reg'] = args.l2_regularizer
    config['grad_clip'] = args.grad_clip
    config['batch_size'] = args.batch_size
    config['kl_weight'] = args.kl_weight
    config['extractor_size'] = args.extractor_size
    config['inference_size'] = args.inference_size
    config['prior_size'] = args.prior_size
    config['rnn_size'] = args.rnn_size
    config['transform_size'] = args.transform_size
    config['reg_weight'] = args.reg_weight
    config['random_seed'] = random_seed
    config['latent_dim'] = args.latent_dim

    # Begin training
    if not dvk_model_train:
        epoch = 0 # We do not train because we will load trained dvk model
    else:
        if logging:
            wandb.init(project = config['env_name'] + '_DVK',
                config = config)

        epoch = train(args, net, x, u, logging, dvk_model_dir, random_seed) ## we get epoch value because train will automatically end when the loss doesnot further decrease

    return net, epoch, args

# Train network
def train(args, net, x, u, logging, dvk_model_dir, random_seed):
    np.random.seed(random_seed)
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        resume_epoch = 0
        # load from previous save
        if len(args.ckpt_name) > 0:
            # "kmixup_seq_len_64_1/" + 'dvk.ckpt-{}'.format(6)
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))
            print(os.path.join(args.save_dir, args.ckpt_name))
            resume_epoch = int(args.ckpt_name[-2:])+1
            print("Resume Epoch: ", resume_epoch)

        # Generate data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        shift_u = sess.run(net.shift_u)
        scale_u = sess.run(net.scale_u)

        # Generate training data
        replay_memory = ReplayMemory(args, shift, scale, shift_u, scale_u,  net, sess, x, u, predict_evolution=True)

        #Function to evaluate loss on validation set
        def val_loss(kl_weight):
            replay_memory.reset_batchptr_val()
            loss = 0.0
            for b in range(replay_memory.n_batches_val):
                # Get inputs
                batch_dict = replay_memory.next_batch_val()
                x = batch_dict['states']
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                feed_in[net.u] = u
                feed_in[net.kl_weight] = kl_weight

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/replay_memory.n_batches_val

        # Specify number of times to loop through training procedure
        n_loops = 10 if args.ilqr else 1
        for n in range(n_loops):
            # Re-initialize network parameters if after the first loop (in place of fine-tuning since training is cheap)
            if n >= 1:
                tf.global_variables_initializer().run()

            # Store normalization parameters
            sess.run(tf.assign(net.shift, replay_memory.shift_x))
            sess.run(tf.assign(net.scale, replay_memory.scale_x))
            sess.run(tf.assign(net.shift_u, replay_memory.shift_u))
            sess.run(tf.assign(net.scale_u, replay_memory.scale_u))

            # Initialize variable to track validation score over time
            old_score = 1e20

            # Set initial learning rate and weight on kl divergence
            print('setting learning rate to ', args.learning_rate)
            sess.run(tf.assign(net.learning_rate, args.learning_rate))
            lr = args.learning_rate

            # Define temperature for annealing kl_weight
            anneal_time = 5
            T = anneal_time*replay_memory.n_batches_train

            # Define counting variables
            count = 0
            count_decay = 0
            decay_epochs = []

            # Loop over epochs
            for e in range(resume_epoch, args.num_epochs):
                # visualize_predictions(args, sess, net, replay_memory, env, e)

                # Initialize loss
                loss = 0.0
                kl_loss = 0.0
                pred_loss = 0.0
                loss_count = 0
                b = 0
                replay_memory.reset_batchptr_train()

                # Loop over batches
                while b < replay_memory.n_batches_train:
                    start = time.time()

                    # Update kl_weight
                    if e < 3:
                        kl_weight = 1e-6
                    else:
                        count += 1
                        kl_weight = min(args.kl_weight, 1e-6 + args.kl_weight*count/float(T))

                    # Get inputs
                    batch_dict = replay_memory.next_batch_train()
                    x = batch_dict['states']
                    u = batch_dict['inputs']

                    # Construct inputs for network
                    feed_in = {}
                    feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                    feed_in[net.u] = u
                    feed_in[net.kl_weight] = kl_weight

                    # Find loss and perform training operation
                    feed_out = [net.cost, net.kl_loss, net.pred_loss, net.train]
                    out = sess.run(feed_out, feed_in)

                    # Update and display cumulative losses
                    loss += out[0]
                    kl_loss += out[1]
                    pred_loss += out[2]
                    loss_count += 1

                    end = time.time()
                    b += 1

                    # Logging
                    if logging:
                        log_dict = dict()
                        log_dict['training_loss'] = out[0]
                        log_dict['kl_loss'] = out[1]
                        log_dict['pred_loss'] = out[2]
                        wandb.log(log_dict)

                    # Print loss
                    if (e * replay_memory.n_batches_train + b) % 1000 == 0 and b > 0:
                        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, loss/loss_count, end - start))
                        print("{}/{} (epoch {}), pred_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, pred_loss/loss_count, end - start))
                        print("{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, kl_loss/loss_count, end - start))

                        print('')
                        loss = 0.0
                        kl_loss = 0.0
                        pred_loss = 0.0
                        loss_count = 0

                # Evaluate loss on validation set
                score = val_loss(kl_weight)
                print('Validation Loss: {0:f}'.format(score))

                # Logging
                if logging:
                    log_dict = dict()
                    log_dict['val_loss'] = score
                    wandb.log(log_dict)

                b = 0

                # Set learning rate
                if (old_score - score) < -0.01 and e >= 8:
                    count_decay += 1
                    decay_epochs.append(e)
                    # if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                    lr = args.learning_rate * (args.decay_rate ** count_decay)
                    if args.ilqr and lr < 5e-5: break
                    print('setting learning rate to ', lr)
                    sess.run(tf.assign(net.learning_rate, lr))
                # Save model every epoch
                checkpoint_path = os.path.join(args.save_dir, dvk_model_dir + args.save_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step = e)
                print("model saved to {}".format(checkpoint_path))

                old_score = score

    return e

if __name__ == '__main__':
    main()
