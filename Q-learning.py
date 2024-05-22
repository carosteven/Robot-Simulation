import numpy as np
import matplotlib.pyplot as plt
import pickle
from maze_env import Maze_Sim

sim = Maze_Sim()
available_act = ['up', 'down', 'left', 'right']
gamma = 0.8
epsilon = 0.1
DEFAULT_ACT = 'up'
train = True
continue_training = False

def init_Q_n(Q, n, s):
    for a in available_act:
        Q[(s, a)] = -1
        n[(s, a)] = 0
    return Q, n

def policy(Q, s):
    # Find argmax_a from Q
    rewards = [Q[(s, i)] for i in available_act]
    best_actions = [available_act[i[0]] for i in np.argwhere(rewards == np.amax(rewards))]
    best_action = np.random.choice(best_actions)
    return best_action

def sample_next_action(Q, s):
    method = np.random.choice(['exploration', 'exploitation'], p=[epsilon, 1-epsilon])

    if method == 'exploration':
        next_action = np.random.choice(available_act)

    elif method == 'exploitation':
        next_action = policy(Q, s)

    return next_action

def get_state():
    position = (round(sim._agent.body.center().x, 0), round(sim._agent.body.center().y, 0))
    velocity = (round(sim._agent.body.velocity.x/50.0)*50, round(sim._agent.body.velocity.y/50.0)*50)

    state = []
    for stat in [position, velocity]:
        for coord in stat:
            state.append(coord)
    return tuple(state)

def train_model(epochs=1, Q={}, n={}):
    initial_state = get_state()
    Q, n = init_Q_n(Q, n, initial_state)

    for epoch in range(epochs):
        score = 0
        sim.__init__()
        # Loop unitl reaches goal
        while sim._running:
            sim.run_controlled()
            
            print(f"Epoch: {epoch}, Score: {score}", end='\r')

            current_state = get_state()
            score = sim._agent.reward

            # Initialize Q and n (if necessary)
            if Q.get((current_state, DEFAULT_ACT)) == None:
                Q, n = init_Q_n(Q, n, current_state)

            # Select action a and execute it
            action = sample_next_action(Q, current_state)
            sim._actions(action)

            # Observe s' and r
            new_state = get_state()
            r = sim._agent.reward - score

            # Initialize Q and n (if necessary)
            if Q.get((new_state, DEFAULT_ACT)) == None:
                Q, n = init_Q_n(Q, n, new_state)

            # Update counts
            n[(current_state, action)] += 1

            # Learning rate
            a = 1 / (n[(current_state, action)])

            # Update Q-value
            next_rewards = [Q[(new_state, i)] for i in available_act]
            Q[(current_state, action)] = Q[(current_state, action)] + a*(r + gamma*max(next_rewards) - Q[(current_state, action)])
        print('\n', end='\r')

        if epoch%10 == 9:
            f = open('Q.pckl', 'wb')
            pickle.dump([Q, n], f)
            f.close()

            print("***************** Test *****************")
            print(f"len(Q): {len(Q)}")
            test_model(Q,n)
            print("\n****************************************\n")

    return [Q, n]

def test_model(Q_actual, n_actual):
    sim.__init__()
    Q = Q_actual
    n = n_actual

    while sim._running:
        sim.run_controlled()
        print(f"Score: {sim._agent.reward}", end='\r')

        current_state = get_state()
        if Q.get((current_state, DEFAULT_ACT)) == None:
                Q, n = init_Q_n(Q, n, current_state)
        
        action = policy(Q, current_state)
        sim._actions(action)


train = False

if train == True:
    if continue_training == True:
        f = open('Q.pckl', 'rb')
        Q, n = pickle.load(f)
        f.close()

    else:
        Q = {}
        n = {}

    Q, n = train_model(50, Q, n)
    f = open('Q.pckl', 'wb')
    pickle.dump([Q, n], f)
    f.close()

else:
    f = open('Q.pckl', 'rb')
    Q, n = pickle.load(f)
    f.close()
    print(len(Q))
    test_model(Q, n)

