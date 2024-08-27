import sys
sys.path.insert(1, './environments')

from nav_obstacle_env import Nav_Obstacle_Env
from push_empty_env import Push_Empty_Env
from push_empty_small_env import Push_Empty_Small_Env

def selector(config):
    if config['environment'] == 0:
        return Nav_Obstacle_Env()
    elif config['environment'] == 1:
        return Push_Empty_Env()
    elif config['environment'] == 2:
        return Push_Empty_Small_Env(config)
    else:
        print("Bad environment selection")
        return None