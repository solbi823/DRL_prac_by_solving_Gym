
import math
import gym
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

DEFAULT_ENV_NAME = "MountainCar-v0"
MEAN_REWARD_BOUND = 10000 * 0.9


class Net(nn.Module):
    def __init__ (self, obs_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),   # 3개의 hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),      # 3개로 하니까 수렴이 잘 안돼서 2개로 줄임
            # nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        x = x.float()                   # 자꾸 더블 텐서 입력한다고 런타임 에러나서 추가
        return self.net(x)


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
        self.max_position = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)

        # reward rule 수정.
        if abs(new_state[0] + (math.pi / 6)) > self.max_position :
            self.max_position = abs(new_state[0] + (math.pi / 6))
            if new_state[0] + (math.pi / 6) > 0:
                reward = self.max_position * 15
            else:
                reward = self.max_position * 10

        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            if self.state[0]>= 0.5:
                self.total_reward += 10000
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":

    env = gym.make(DEFAULT_ENV_NAME)

    print(env.observation_space.shape)
    print(env.action_space)

    input_data = env.observation_space.shape[0]
    net = Net(input_data, env.action_space.n)
    tgt_net = Net(input_data, env.action_space.n)
    writer = SummaryWriter("MountainCar-v0")
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            # writer.add_scalar("epsilon", epsilon, frame_idx)
            # writer.add_scalar("speed", speed, frame_idx)
            # writer.add_scalar("reward_100", mean_reward, frame_idx)
            # writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net)
        loss_t.backward()
        optimizer.step()
    writer.close()
