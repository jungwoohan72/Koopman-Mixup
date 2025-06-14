from .. import offline_env
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv, SwimmerEnv, InvertedPendulumEnv, InvertedDoublePendulumEnv, ReacherEnv
from ..utils.wrappers import NormalizedBoxEnv

class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineSwimmerEnv(SwimmerEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SwimmerEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineReacherEnv(ReacherEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        ReacherEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))

def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))

def get_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))

def get_swimmer_env(**kwargs):
    return NormalizedBoxEnv(OfflineSwimmerEnv(**kwargs))

def get_reacher_env(**kwargs):
    return NormalizedBoxEnv(OfflineReacherEnv(**kwargs))      

if __name__ == '__main__':
    """Example usage of these envs"""
    pass
