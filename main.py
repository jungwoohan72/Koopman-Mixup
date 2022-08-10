import argparse
import yaml
import d3rlpy
import importlib, sys
import argparse

import numpy as np

from d3rlpy.metrics.scorer import evaluate_on_environment
from koopman_data_aug import mixup_data_aug, koop_mixup_data_aug
from baseline_data_aug import rad_data_aug, s4rl_data_aug, nmer_data_aug

def main(args,
		model_type="CQL",
		env_mode="hopper-medium-v0",
		process="koop_aug",
		dvk_model_train=False, # If True, one shot dvk train + CQL train. Else, you should specify the target dvk_model_dir.
		use_all=False, # If True, not concatenating the trajectory to trial_len. Instead, use all. 
		random_seed = 1,
		dvk_model_dir='',
		trial_len=768,
		use_gpu=True):
	
	# Specify randomo seed
	np.random.seed(random_seed)

	# Load offline dataset and environment
	if env_mode.split("-")[0] in ["hopper", "halfcheetah", "walker"]:
		dataset, env = d3rlpy.datasets.get_dataset(env_mode)
	else:
		dataset, env = d3rlpy.datasets.get_d4rl(env_mode) # For Swimmer and Reacher experiments

	# Load yaml file containing hyperparameters for CQL
	with open("reproductions.yaml") as p:
		parameters = yaml.load(p, Loader=yaml.FullLoader)

	config = parameters[model_type]

	encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units = args.CQL_size)
	config["env_name"] = str(env).split()[1].lstrip("<")
	# config["env_name"] = str(env).split("<")[2].rstrip(">")
	config["actor_encoder_factory"] = encoder
	config["critic_encoder_factory"] = encoder

	args.CQL_batch_size = config["batch_size"]

	# Model Initialization
	if model_type == "CQL":
		model = d3rlpy.algos.CQL(actor_learning_rate=config['actor_learning_rate'],
								 critic_learning_rate=config['critic_learning_rate'],
								 temp_learning_rate=config['temp_learning_rate'],
								 actor_encoder_factory=config['actor_encoder_factory'],
								 critic_encoder_factory=config['critic_encoder_factory'],
								 batch_size=config['batch_size'],
								 n_action_samples=config['n_action_samples'],
								 alpha_learning_rate=config['alpha_learning_rate'],
								 conservative_weight=config['conservative_weight'],
								 use_gpu = use_gpu)

	evaluate_scorer = evaluate_on_environment(env, n_trials=10) # Evaluation metric that deploys the learned policy in the environment and obtain the average return obtained from 10 evaluations.

	# Original offline RL without data augmentation
	if process == "normal":
		config['type'] = 'normal'

		model.build_with_dataset(dataset)

		n_epochs = int(1000000/(len(dataset.observations)/args.batch_size)) # n_epochs to satisfy 1M gradient update steps

		# Training model
		model.fit(dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
				'environment': d3rlpy.metrics.evaluate_on_environment(env),
				'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_normal'.format(env_mode))
		
		# Evaluation after the training
		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

	# Using original mixup
	if process=="aug":
		config['type'] = 'mixup'

		# Pre-processing to create the augmented dataset that includes newly created data uisng mixup
		new_dataset, n_epochs = mixup_data_aug(args, 
											env, 
											dataset, 
											trial_len, 
											random_seed,
											n_aug=int(len(dataset.rewards)*args.n_aug))

		model.build_with_dataset(new_dataset)

		# Training model
		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
				'environment': d3rlpy.metrics.evaluate_on_environment(env),
				'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_aug'.format(env_mode))

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)		

	## using k-mixup augmented data
	if process=="koop_aug":
		config['type'] = 'kmixup'

		# Pre-processing to created the augmented dataset that includes newly created data using k-mixup
		new_dataset, n_epochs, _,_,_ = koop_mixup_data_aug(args,
												env,
												dataset,
												dvk_model_train = dvk_model_train,
												n_aug=int(len(dataset.rewards)*args.n_aug),
												use_all=use_all,
												dvk_model_dir=dvk_model_dir,
												trial_len=trial_len,
												random_seed=random_seed)

		model.build_with_dataset(new_dataset)

		# Model Training
		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
					'environment': d3rlpy.metrics.evaluate_on_environment(env),
					'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_dvk_aug'.format(env_mode))

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

	## using rad augmented data
	if process=="rad":
		config['type'] = 'rad'

		new_dataset, n_epochs = rad_data_aug(args,
										dataset,
										n_aug=int(len(dataset.rewards)*args.n_aug),
										random_seed=random_seed)


		model.build_with_dataset(new_dataset)

		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
					'environment': d3rlpy.metrics.evaluate_on_environment(env),
					'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_dvk_aug'.format(env_mode))

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

	## using s4rl augmented data
	if process=="s4rl":
		config['type'] = 's4rl'

		new_dataset, n_epochs = s4rl_data_aug(args,
										dataset,
										n_aug=int(len(dataset.rewards)*args.n_aug),
										random_seed=random_seed)


		model.build_with_dataset(new_dataset)

		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
					'environment': d3rlpy.metrics.evaluate_on_environment(env),
					'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_dvk_aug'.format(env_mode))

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

	## using nmer augmented data
	if process=="nmer":
		config['type'] = 'nmer'

		new_dataset, n_epochs = nmer_data_aug(args,
										dataset,
										n_aug=int(len(dataset.rewards)*args.n_aug),
										random_seed=random_seed)

		model.build_with_dataset(new_dataset)

		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
					'environment': d3rlpy.metrics.evaluate_on_environment(env),
					'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_dvk_aug'.format(env_mode))

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)


if __name__ == '__main__':

	###################################################################################### Hyperparameters for each dataset ######################################################################################

						## hopper-medium-v0:             trial_len: 768 / seqlen: 64 / Subseq: 30k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-v0_512256256_512256_512_40
						## walker2d-medium-v0:           trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / walker2d-medium-v0_512256256_512256_512_60_0.0005

						## hopper-medium-expert-v0:      trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-expert-v0_512256256_512256_512_40
						## walker2d-medium-expert-v0:    trial_len: 990 / seqlen: 64 / Subseq: 50k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0005 / walker2d-medium-expert-v0_512256256_512256_512_60

						## hopper-medium-replay-v0:      trial_len: 300 / seqlen: 64 / Subseq: 34k / Aug: 0.3 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-replay-v0_512256256_512256_512_40
						## walker2d-medium-replay-v0:    trial_len: 300 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / walker2d-medium-replay-v0_512256256_512256_512_50

						## swimmer-medium-v0:            trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0005 / CQL: 256 / swimmer-medium-v0_128128_12864_128_10
						## reacher-medium-v0:            trial_len: 50 / seqlen: 24 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / CQL: 256 / reacher-medium-v0_128128_12864_128_15

	##############################################################################################################################################################################################################

	######################################
	#             Main Params            #
	######################################
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type',     type=str,   default='CQL',                                       help='RL algorithm')
	parser.add_argument('--env',            type=str,   default='hopper-medium-v0',                          help='environment')
	parser.add_argument('--process',        type=str,   default='koop_aug',                                  help='normal for naive baseline, aug for mixup, koop_aug for k-mixup, rad for RAD, s4rl for S4RL, nmer for NMER')
	parser.add_argument('--dvk_train',      type=bool,  default=False,                                       help='set true to train a new dvk model')
	parser.add_argument('--use_all',        type=bool,  default=False,                                       help='to use all the trajectories longer than trial_len')
	parser.add_argument('--seed',           type=int,   default=0,                                           help='random seed for sampling operations')
	parser.add_argument('--dvk_dir',        type=str,   default='hopper-medium-v0_512256256_512256_512_40/', help='dvk model directory which is in model folder')
	parser.add_argument('--trial_len',      type=int,   default=768,                                         help='lower limit for trajectory length that will be used for the training')
	parser.add_argument('--n_subseq',       type=int,   default=34000,                                       help='lower limit for the number of subsequences that will be used for the DVK training')
	parser.add_argument('--use_gpu',        type=bool,  default=True, 										 help='set true to use GPU')

	######################################
	#             DVK Params             #
	######################################
	parser.add_argument('--save_dir',       type=str,   default='./models',    								 help='directory to store models')
	parser.add_argument('--val_frac',       type=float, default=0.1,                						 help='fraction of data to be witheld in validation set')
	parser.add_argument('--ckpt_name',      type=str,   default="",                 						 help='name of model file to load (blank means none)')
	parser.add_argument('--save_name',      type=str,   default='dvk',              						 help='name of model files for saving')
	parser.add_argument('--domain_name',    type= str,  default='custom',           						 help='environment name')

	parser.add_argument('--seq_length',     type=int,   default= 64,                						 help='subsequence length for the training')
	parser.add_argument('--mpc_horizon',    type=int,   default= 16,                						 help='horizon to consider for MPC') # Not used for K-mixup
	parser.add_argument('--batch_size',     type=int,   default= 32,                						 help='minibatch size') 
	parser.add_argument('--latent_dim',     type=int,   default= 40,                						 help='dimensionality of the Koopman invariant subspace') 
	parser.add_argument('--n_aug',          type=int,   default= 0.2,               						 help='proportion of dataset that would be augmented')

	parser.add_argument('--num_epochs',     type=int,   default= 60,                						 help='number of epochs') 
	parser.add_argument('--learning_rate',  type=float, default= 0.0005,            						 help='learning rate')
	parser.add_argument('--decay_rate',     type=float, default= 0.5,               						 help='decay rate for learning rate')
	parser.add_argument('--l2_regularizer', type=float, default= 1.0,               						 help='regularization for least squares')
	parser.add_argument('--grad_clip',      type=float, default= 5.0,              							 help='clip gradients at this value')
	parser.add_argument('--kl_weight',      type=float, default= 0,       									 help='weight applied to kl-divergence loss')
	
	######################################
	#          Network Params            #
	######################################
	parser.add_argument('--CQL_size',       nargs='+', type=int, default=[256, 256, 256],    				 help='hidden layer sizes for CQL')
	parser.add_argument('--extractor_size', nargs='+', type=int, default=[512. 256, 256],    				 help='hidden layer sizes in feature extractor/decoder')
	parser.add_argument('--inference_size', nargs='+', type=int, default=[512, 256, 256],    				 help='hidden layer sizes in feature inference network')
	parser.add_argument('--prior_size',     nargs='+', type=int, default=[512, 256],         				 help='hidden layer sizes in prior network')
	parser.add_argument('--rnn_size',       type=int,   default= 512,                        				 help='size of RNN layers')
	parser.add_argument('--transform_size', type=int,   default= 512,                        				 help='size of transform layers')
	parser.add_argument('--reg_weight',     type=float, default= 1e-4,                       				 help='weight applied to regularization losses')

	#####################################
	#       Addtitional Options         #
	#####################################
	parser.add_argument('--ilqr',           type=bool,  default= False,     								 help='whether to perform ilqr with the trained model') # Not used for K-mixup
	parser.add_argument('--evaluate',       type=bool,  default= False,     								 help='whether to evaluate trained network') # Not used for K-mixup
	parser.add_argument('--perform_mpc',    type=bool,  default= False,    									 help='whether to perform MPC instead of training') # Not used for K-mixup
	parser.add_argument('--worst_case',     type=bool,  default= False,     								 help='whether to optimize for worst-case cost') # Not used for K-mixup
	parser.add_argument('--gamma',          type=float, default= 1.0,       								 help='discount factor') 
	parser.add_argument('--num_models',     type=int,   default= 5,         								 help='number of models to use in MPC') # Not used for K-mixup
	parser.add_argument('--alpha',          type=float, default= 0.2,       								 help='number of models to use in MPC') # Not used for K-mixup

	#####################################
	#        For Other Data Aug         #
	#####################################
	parser.add_argument('--additional',    	type=bool,  default= True,     									 help='adding additional augmented data or not')

	#####################################
	#         For RAD Data Aug          #
	#####################################
	parser.add_argument('--single_flag',    type=bool,  default= False,    									 help='RAS-S for False / RAS-M for True for RAD data augmentation')
	parser.add_argument('--scale',    		type=bool,  default= True,     								     help='RAS-S or RAS-M for True / Gaussian noise addition for False')

	#####################################
	#        For S4RL Data Aug          #
	#####################################
	parser.add_argument('--gaussian',    	type=bool,  default= False,    									 help='Gaussian noise addition for True / Uniform noise adddition for False')

	args = parser.parse_args()
	
	main(args,
		model_type=args.model_type,
		env_mode=args.env,
		process=args.process,
		dvk_model_train=args.dvk_train,
		use_all=args.use_all,
		random_seed=args.seed,
		dvk_model_dir=args.dvk_dir,
		trial_len=args.trial_len,
		use_gpu=args.use_gpu)