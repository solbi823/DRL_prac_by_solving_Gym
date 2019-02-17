import numpy as np
import torch
import torch.nn as nn
import time
import sys

# hyperparameters for Pong env. stored in dictionary
HYPERPARAMS = {
        'acrobot' : {
            'stop_reward':      0.0,
            'run_name':         'acrobot',
            'env_name':         "Acrobot-v1",
            'replay_size':      100000,
            'replay_initial':   10000,
            'target_net_sync':  1000,
            'epsilon_frames':   10**5,
            'epsilon_start':    1.0,
            'epsilon_final':    0.02,
            'learning_rate':    0.0001,
            'gamma':            0.99,
            'batch_size':       32
            },
        }


# batch of transitions --> pack into a set of NumPy arrays
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

def calc_loss_dqn(batch, net, tgt_net, gamma, device="CPU"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
                max(self.epsilon_final, self.epsilon_start - frame/self.epsilon_frames)

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" %epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f/s%s" %(\
                frame, len(self.total_rewards), mean_reward, speed, epsilon_str)
            )
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
            self.writer.add_scalar("speed", speed, frame)
            self.writer.add_scalar("reward_100", mean_reward, frame)
            self.writer.add_scalar("reward", reward, frame)
            if mean_reward > self.stop_reward:
                print("Solved in %d frames!" %frame)
                return True
        return False

