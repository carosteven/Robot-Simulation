import sys
sys.path.insert(1, './environments')

from nav_obstacle_env import Nav_Obstacle_Env
from push_empty_env import Push_Empty_Env
from push_empty_small_env import Push_Empty_Small_Env

def selector(env_num):
    if env_num == 0:
        return Nav_Obstacle_Env()
    elif env_num == 1:
        return Push_Empty_Env()
    elif env_num == 2:
        return Push_Empty_Small_Env()
    else:
        print("Bad environment selection")
        return None