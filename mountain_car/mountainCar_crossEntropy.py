import math
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
# iteration 마다 실행하는 episodes 수 
BATCH_SIZE = 100	
# reward boundary (30퍼만 남김)
PERCENTILE = 70

class Net(nn.Module):
	def __init__ (self, obs_size, hidden_size, n_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
			)
	def forward(self, x):
		return self.net(x)


Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation','action'])

def iterate_batches(env, net, batch_size):

	# batch 리스트 안에 episode 들이 들어가게 됨.
	batch = []

	max_position = 0.0

	# current episode reward
	episode_reward = 0.0

	# current episode steps
	episode_steps = []
	obs = env.reset()
	sm = nn.Softmax(dim = 1)

	while True:
		obs_v = torch.FloatTensor([obs])
		act_probs_v = sm(net(obs_v))
		act_probs = act_probs_v.data.numpy()[0]

		# 이 확률로 action 선택한다. 
		action = np.random.choice(len(act_probs), p = act_probs)
		next_obs, reward, is_done, _ = env.step(action)

		if abs(next_obs[0] + (math.pi / 6)) > max_position :
			max_position = abs(next_obs[0] + (math.pi / 6))
			if next_obs[0] + (math.pi / 6) > 0:
				reward = max_position * 15
			else:
				reward = max_position * 10

		episode_reward += reward
		episode_steps.append(EpisodeStep(observation = obs, action = action))

		if is_done:

			if next_obs[0] >= 0.5:
				episode_reward = 10000

			batch.append(Episode(reward = episode_reward, steps = episode_steps))

			max_position = 0.0
			episode_reward = 0.0
			episode_steps = []
			next_obs = env.reset()
			if len(batch) == batch_size:
				yield batch
				batch = []

		obs = next_obs

def filter_batch(batch, percentile):
	rewards = list(map(lambda s : s.reward, batch))
	reward_bound = np.percentile(rewards, percentile)
	reward_mean = float(np.mean(rewards))

	train_obs = []
	train_act = []
	elite_batch = []

	for example in batch:
		if example.reward < reward_bound:
			continue
		# iterable 객체의 elements 를 append
		train_obs.extend(map(lambda step : step.observation, example.steps))
		train_act.extend(map(lambda step : step.action, example.steps))
		elite_batch.append(example)

	train_obs_v = torch.FloatTensor(train_obs)
	train_act_v = torch.LongTensor(train_act)
	return elite_batch, train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
	env = gym.make("MountainCar-v0")
	env = gym.wrappers.Monitor(env, directory="mon", force=True)
	obs_size = env.observation_space.shape[0]
	n_actions = env.action_space.n
	print(obs_size)

	net = Net(obs_size, HIDDEN_SIZE, n_actions)
	# loss function(type of objective function)
	objective = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params = net.parameters(), lr = 0.01)
	writer = SummaryWriter(comment="-mountainCar")

	full_batch = []
	# enumerate은 count와 리스트 element 를 전달
	for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

		# 각각의 batch에 대해 elite episodes 의 data 만 받아 온다.
		full_batch, obs_v, acts_v, reward_b, reward_m = filter_batch(full_batch + batch, PERCENTILE)
		if not full_batch:
			continue
		full_batch = full_batch[-5:]

		optimizer.zero_grad()
		# filtered data 가운데 observations를 네트워크에 input 한다.
		action_scores_v = net(obs_v)
		# output 과 desired output을 손실 함수에 넣는다.
		# desired output: elite episode data 가운데 observation과 짝을 이루는 action
		loss_v = objective(action_scores_v, acts_v)
		# loss value 를 back propagation 후 weight 갱신.
		loss_v.backward()
		optimizer.step()

		print("%d: loss = %.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
		writer.add_scalar("loss", loss_v.item(), iter_no)
		writer.add_scalar("reward_bound", reward_b, iter_no)
		writer.add_scalar("reward_mean", reward_m, iter_no)

		# 90퍼 이상 성공.
		if reward_m > 10000 * (BATCH_SIZE - 10)/ BATCH_SIZE:	
			print("Solved!")
			break

	writer.close()
