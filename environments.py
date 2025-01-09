import sys
sys.path.insert(1, './environments')


def selector(config):
    if config['environment'] == 0:
        from nav_obstacle_env import Nav_Obstacle_Env
        return Nav_Obstacle_Env()
    elif config['environment'] == 1:
        from push_empty_env import Push_Empty_Env
        return Push_Empty_Env()
    elif config['environment'] == 2:
        from push_empty_small_env import Push_Empty_Small_Env
        return Push_Empty_Small_Env(config)
    elif config['environment'] == 3:
        from basic_env import Basic_Env
        return Basic_Env(config)
    elif config['environment'] == 4:
        return get_env_from_cfg(config, use_gui=config['use_gui'])
    else:
        print("Bad environment selection")
        return None
    
def get_env_from_cfg(cfg, **kwargs):
    from pb_env import PB_Env
    kwarg_list = [
        'room_length', 'room_width', 'num_cubes', 'obstacle_config',
        'use_distance_to_receptacle_channel', 'distance_to_receptacle_channel_scale',
        'use_shortest_path_to_receptacle_channel', 'use_shortest_path_channel', 'shortest_path_channel_scale',
        'use_position_channel', 'position_channel_scale',
        'partial_rewards_scale', 'use_shortest_path_partial_rewards', 'collision_penalty', 'nonmovement_penalty',
        'use_shortest_path_movement', 'fixed_step_size', 'use_steering_commands', 'steering_commands_num_turns',
        'ministep_size', 'inactivity_cutoff', 'random_seed',
    ]
    original_kwargs = {}
    for kwarg_name in kwarg_list:
        original_kwargs[kwarg_name] = cfg[kwarg_name]
    original_kwargs.update(kwargs)
    return PB_Env(**original_kwargs)