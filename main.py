import argparse
import yaml
import d3rlpy
import importlib, sys
import wandb
import argparse

import numpy as np

from d3rlpy.metrics.scorer import evaluate_on_environment
from koopman_data_aug import mixup_data_aug, koop_mixup_data_aug

def main(args,
		model_type="CQL",
		env_mode="hopper-medium-v0",
		process="koop_aug",
		dvk_model_train=False, # If True, one shot dvk train -> CQL train. Else, you should specify dvk_model_dir.
		use_all=False, # If True, not concatenating the trajectory to trial_len. Instead, use all.
		logging=False, # If you don't use wandb, set it False. Otherwise, there would be error.
		random_seed = 1,
		dvk_model_dir='',
		trial_len=768,
		use_gpu=True):
	
	np.random.seed(random_seed)

	# Load offline dataset and environment
	if env_mode.split("-")[0] in ["hopper", "halfcheetah", "walker"]:
		dataset, env = d3rlpy.datasets.get_dataset(env_mode)
	else:
		dataset, env = d3rlpy.datasets.get_d4rl(env_mode)

	# Hyperparameters for CQL
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
	# elif model_type == "BEAR":
	# 	model = d3rlpy.algos.BEAR(use_gpu = True)
	# elif model_type == "AWAClusBC":
	# 	model = d3rlpy.algos.AWAClusBC(use_gpu = True)
	# else:
	# 	model = d3rlpy.algos.IQL(use_gpu = True)

	evaluate_scorer = evaluate_on_environment(env, n_trials=10) # evaluation metric

	## original offline RL
	if process == "normal":
		config['type'] = 'normal'

		model.build_with_dataset(dataset)

		if logging:
			wandb.init(project=config['env_name']+"_"+model_type,
				   config=config)

		n_epochs = int(1000000/(len(dataset.observations)/args.batch_size)) # n_epochs to satisfy 1M gradient update steps

		model.fit(dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
				'environment': d3rlpy.metrics.evaluate_on_environment(env),
				'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_normal'.format(env_mode),
				wandb_log = logging)

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

	## using original mixup augmented dataset
	if process=="aug":
		config['type'] = 'mixup'

		new_dataset, n_epochs = mixup_data_aug(args, 
											env, 
											dataset, 
											trial_len, 
											random_seed,
											n_aug=int(len(dataset.rewards)*args.n_aug))

		model.build_with_dataset(new_dataset)

		if logging:
			wandb.init(project=config['env_name']+"_"+model_type,
					   config=config)

		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
				'environment': d3rlpy.metrics.evaluate_on_environment(env),
				'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_aug'.format(env_mode),
				wandb_log=logging)

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)		

	## using k-mixup augmented data
	if process=="koop_aug":
		config['type'] = 'kmixup'

		new_dataset, n_epochs, _,_,_ = koop_mixup_data_aug(args,
												env,
												dataset,
												dvk_model_train = dvk_model_train,
												n_aug=int(len(dataset.rewards)*args.n_aug),
												use_all=use_all,
												logging=logging,
												dvk_model_dir=dvk_model_dir,
												trial_len=trial_len,
												random_seed=random_seed)

		model.build_with_dataset(new_dataset)

		if logging:
			wandb.init(project=config['env_name']+"_"+model_type,
					   config=config)

		model.fit(new_dataset,
				eval_episodes = dataset,
				n_epochs=n_epochs,
				scorers={
					'environment': d3rlpy.metrics.evaluate_on_environment(env),
					'td_error': d3rlpy.metrics.td_error_scorer
				},
				logdir='custom_logs',
				experiment_name='{}_dvk_aug'.format(env_mode),
				wandb_log = logging)

		mean_episode_return = evaluate_scorer(model)
		print(mean_episode_return)

if __name__ == '__main__':

	## halfcheetah-medium-v0:        trial_len: 990 / seqlen: 64 / subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / halfcheetah-medium-v0_256256_256_128_50
	## hopper-medium-v0:             trial_len: 768 / seqlen: 64 / Subseq: 30k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-v0_512256256_512256_512_40
	## walker2d-medium-v0:           trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / walker2d-medium-v0_512256256_512256_512_60_0.0005

	## halfcheetah-medium-expert-v0: trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.1 / Temp: 0.0001 / halfcheetah-medium-expert-v0_512256256_512256_512_60
	## hopper-medium-expert-v0:      trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-expert-v0_512256256_512256_512_40
	## walker2d-medium-expert-v0:    trial_len: 990 / seqlen: 64 / Subseq: 50k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0005 / walker2d-medium-expert-v0_512256256_512256_512_60

	## halfcheetah-medium-replay-v0: trial_len: 990 / seqlen: 64 / Subseq: 100k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / halfcheetah-medium-replay-v0_512256256_512256_512_60
	## hopper-medium-replay-v0:      trial_len: 300 / seqlen: 64 / Subseq: 34k / Aug: 0.3 / Alpha: 0.2 / Temp: 0.0001 / hopper-medium-replay-v0_512256256_512256_512_40
	## walker2d-medium-replay-v0:    trial_len: 300 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / walker2d-medium-replay-v0_512256256_512256_512_50

	## inverted-pendulum-random-v0:  trial_len: 51 / seqlen: 25 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / CQL: 256 / invertedpendulum-random-v1_6464_6432_64_5
	## swimmer-medium-v0:            trial_len: 990 / seqlen: 64 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0005 / CQL: 256 / swimmer-medium-v0_128128_12864_128_10
	## reacher-medium-v0:            trial_len: 50 / seqlen: 24 / Subseq: 34k / Aug: 0.2 / Alpha: 0.2 / Temp: 0.0001 / CQL: 256 / reacher-medium-v0_128128_12864_128_15

	######################################
	#             Main Params            #
	######################################
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type',     type=str,   default='CQL',                                       help='RL algorithm')
	parser.add_argument('--env',            type=str,   default='hopper-medium-v0',                          help='environment')
	parser.add_argument('--process',        type=str,   default='koop_aug',                                  help='normal for naive baseline, aug for mixup, koop_aug for k-mixup')
	parser.add_argument('--dvk_train',      type=bool,  default=False,                                        help='set true to train a new dvk model')
	parser.add_argument('--use_all',        type=bool,  default=False,                                       help='to use all the trajectories longer than trial_len')
	parser.add_argument('--logging',        type=bool,  default=False,                                       help='set true if you want to log via wandb')
	parser.add_argument('--seed',           type=int,   default=0,                                           help='random seed for sampling operations')
	parser.add_argument('--dvk_dir',        type=str,   default='hopper-medium-v0_512256256_512256_512_40/', help='dvk model directory which is in checkpoints folder')
	parser.add_argument('--trial_len',      type=int,   default=768,                                         help='lower limit for trajectory length that will be used for the training')
	parser.add_argument('--n_subseq',       type=int,   default=34000,                                       help='lower limit for trajectory length that will be used for the training')
	parser.add_argument('--use_gpu',        type=bool,  default=False, 										 help='set true to use GPU')

	######################################
	#             DVK Params             #
	######################################
	parser.add_argument('--save_dir',       type=str,   default='./checkpoints',    help='directory to store checkpointed models')
	parser.add_argument('--val_frac',       type=float, default=0.1,                help='fraction of data to be witheld in validation set')
	parser.add_argument('--ckpt_name',      type= str,  default="",                 help='name of checkpoint file to load (blank means none)')
	parser.add_argument('--save_name',      type= str,  default='dvk',              help='name of checkpoint files for saving')
	parser.add_argument('--domain_name',    type= str,  default='custom',           help='environment name')

	parser.add_argument('--seq_length',     type=int,   default= 64,                help='sequence length for training')
	parser.add_argument('--mpc_horizon',    type=int,   default= 16,                help='horizon to consider for MPC')
	parser.add_argument('--batch_size',     type=int,   default= 32,                help='minibatch size')
	parser.add_argument('--latent_dim',     type=int,   default= 40,                help='dimensionality of code')
	parser.add_argument('--n_aug',          type=int,   default= 0.2,               help='proportion of augmented data')

	parser.add_argument('--num_epochs',     type=int,   default= 60,                help='number of epochs')
	parser.add_argument('--learning_rate',  type=float, default= 0.0005,            help='learning rate')
	parser.add_argument('--decay_rate',     type=float, default= 0.5,               help='decay rate for learning rate')
	parser.add_argument('--l2_regularizer', type=float, default= 1.0,               help='regularization for least squares')
	parser.add_argument('--grad_clip',      type=float, default= 5.0,               help='clip gradients at this value')
	parser.add_argument('--kl_weight',      type=float, default= 0,       			help='weight applied to kl-divergence loss')
	
	######################################
	#          Network Params            #
	######################################
	parser.add_argument('--CQL_size',       nargs='+', type=int, default=[256, 256, 256],    help='hidden layer sizes for CQL')
	parser.add_argument('--extractor_size', nargs='+', type=int, default=[512, 256, 256],    help='hidden layer sizes in feature extractor/decoder')
	parser.add_argument('--inference_size', nargs='+', type=int, default=[512, 256, 256],    help='hidden layer sizes in feature inference network')
	parser.add_argument('--prior_size',     nargs='+', type=int, default=[512, 256],         help='hidden layer sizes in prior network')
	parser.add_argument('--rnn_size',       type=int,   default= 512,                        help='size of RNN layers')
	parser.add_argument('--transform_size', type=int,   default= 512,                        help='size of transform layers')
	parser.add_argument('--reg_weight',     type=float, default= 1e-4,                       help='weight applied to regularization losses')

	#####################################
	#       Addtitional Options         #
	#####################################
	parser.add_argument('--ilqr',           type=bool,  default= False,     help='whether to perform ilqr with the trained model')
	parser.add_argument('--evaluate',       type=bool,  default= False,     help='whether to evaluate trained network')
	parser.add_argument('--perform_mpc',    type=bool,  default= False,     help='whether to perform MPC instead of training')
	parser.add_argument('--worst_case',     type=bool,  default= False,     help='whether to optimize for worst-case cost')
	parser.add_argument('--gamma',          type=float, default= 1.0,       help='discount factor')
	parser.add_argument('--num_models',     type=int,   default= 5,         help='number of models to use in MPC')
	parser.add_argument('--alpha',          type=float, default= 0.2,         help='number of models to use in MPC')

	args = parser.parse_args()
	
	main(args,
		model_type=args.model_type,
		env_mode=args.env,
		process=args.process,
		dvk_model_train=args.dvk_train,
		use_all=args.use_all,
		logging=args.logging,
		random_seed=args.seed,
		dvk_model_dir=args.dvk_dir,
		trial_len=args.trial_len,
		use_gpu=args.use_gpu)