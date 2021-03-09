import numpy as np
import matplotlib.pyplot as plt

# 自作クラス
from Brain import Brain


class MazeSarsaBrain(Brain):
	"""
	迷宮問題のBrain
	[public]

	[protected]
		_brain_parameters:list <float> / shape=[n_states, n_actions]
			行動方策πを決定するためのパラメータθ
			※ 行動方策を表敬式で実装するために、対応するパラメータも表形式で実装する
			※ 進行方向に壁があって進めない様子を実装するために、進めない方向には'np.nan'で初期化
			※ 状態s8は、ゴール状態のためパラメータは定義しない
		_policy：list <float> / shape=[n_states, n_actions]

	[private] 

	"""
	def __init__(self, n_states, n_actions, brain_parameters, epsilon=0.5, gamma=0.9, learning_rate=0.1):
		super().__init__(n_states, n_actions)
		self._brain_parameters = brain_parameters
		self._policy = np.zeros(shape=(self._n_states, self._n_actions))
		self._policy = self.convert_into_policy_from_brain_parameters(self._brain_parameters)
		self._q_function = self.init_q_function(brain_parameters=self._brain_parameters)
		self._epsilon = epsilon
		self._gamma = gamma
		self._learning_rate = learning_rate
		return

	def print(self, str):
		print("----------------------------------")
		print("MazeSarsaBrain")
		print(self)
		print(str)
		print("_n_states : ", self._n_states)
		print("_n_actions : ", self._n_actions)
		print("_policy :\n", self._policy)
		print("_brain_parameters :\n", self._brain_parameters)
		print("_q_function :\n", self._q_function)
		print("_epsilon : ", self._epsilon)
		print("_gamma : ", self._gamma)
		print("_learning_rate : ", self._learning_rate)
		print("----------------------------------")
		return

	def get_q_function(self):
		"""
		Q関数の値を取得する
		"""
		return self._q_function

	def decay_epsilon(self):
		"""
		ε-Greedy法のεを減衰する
		"""
		self._epsilon = self._epsilon / 2.0
		return

	def decay_learning_rate(self):
		"""
		学習率を減衰
		"""
		self._learning_rate /= 2.0
		return

	def convert_into_policy_from_brain_parameters(self, brain_parameters):
		"""
		方策パラメータから行動方策を決定する
		"""
		[m, n] = brain_parameters.shape
		policy = np.zeros(shape=(m, n))
		for i in range(m):
			# 割合の計算
			policy[i, :] = brain_parameters[i, :] / np.nansum(brain_parameters[i, :]) # nansumはnamを除いたsum

		# Nan値は0に変換
		policy = np.nan_to_num(policy)

		return policy

	def action(self, state):
		"""
		状態ｓでの行動ａを決定
		[Args]
			state：<int> 現在の状態
		"""
		# ε-Greedyに従った行動選択
		if self._epsilon >= np.random.rand():
			# 探索
			action = np.random.choice(self._n_actions, p=self._policy[state, :])
		else:
			# 活用
			action = np.nanargmax(self._q_function[state, :]) # nanargmaxはnanを無視する

		if action == np.nan:
			action = 0

		return action

	def init_q_function(self, brain_parameters):
		"""
		Q関数を表形式で初期化
		"""
		# エピソードの開始時点でランダムに初期化
		# brain_parametersをかけることで、壁方向はnp.nanとなる
		[a, b] = brain_parameters.shape
		q_function = np.random.rand(a, b) * brain_parameters
		return q_function

	def update_q_function(self, state, action, next_state, next_action, reward):
		"""
		Q値の更新

		[Args]
			state：<int> 現在の状態ｓ
			action：<int> 現在の行動ａ
			next_state：<int> 次の状態ｓ'
			next_action：<int> 次の行動ａ'
			reward：<float> 報酬

		[Returns]

		"""

		if next_state == 8:
			self._q_function[state, action] += self._learning_rate * (reward - self._q_function[state, action])
		else:
			self._q_function[state, action] += self._learning_rate * (reward + self._gamma*self._q_function[next_state, next_action] - self._q_function[state, action])

		return self._q_function