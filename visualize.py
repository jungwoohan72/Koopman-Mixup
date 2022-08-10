import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
import gym
import argparse

def main(args):
    env_mode = args.env
    env = gym.make(env_mode)

    env.seed(1)

    evaluate_scorer = evaluate_on_environment(env, n_trials=100, render=True)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory(hidden_units = args.CQL_size)

    cql = d3rlpy.algos.CQL(actor_encoder_factory=encoder,
                           critic_encoder_factory=encoder,
                           use_gpu = True)

    cql.build_with_env(env)
    cql.load_model('./custom_logs/' + args.CQL_dir + args.CQL_file)

    mean_episode_return = evaluate_scorer(cql)
    print(mean_episode_return)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',            type=str,   default='hopper-medium-v0',                          help='environment')
    parser.add_argument('--CQL_size',       nargs='+',  type=int, default=[256, 256, 256],                   help='hidden layer sizes for CQL')
    parser.add_argument('--CQL_dir',        type=str,   default='reacher-medium-v0_normal_20220311214115/',  help='CQL model directory')
    parser.add_argument('--CQL_file',       type=str,   default='model_144522.pt',                           help='CQL model file')

    args = parser.parse_args()

    main(args)