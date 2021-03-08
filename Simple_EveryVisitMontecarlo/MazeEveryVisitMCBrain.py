import numpy as np
import matplotlib.pyplot as plt

# 自作クラス
from Brain import Brain


class MazeEveryVisitMCBrain(Brain):
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
	def __init__(self, n_states, n_actions, brain_parameters, epsilon=0.5, gamma=0.9):
		super().__init__(n_states, n_actions)
		self._brain_parameters = brain_parameters
		self._policy = np.zeros(shape=(self._n_states, self._n_actions))
		self._policy = ***
		self._q_function = ***
		self._epsilon = epsilon
		self._gamma = gamma
		return

	def print(self, str):
		print("----------------------------------")
		print("MazeRamdomBrain")
		print(self)
		print(str)
		print("_n_states : \n", self._n_states)
		print("_n_actions : \n", self._n_actions)
		print("_policy : \n", self._policy)
		print("_brain_parameters : \n", self._brain_parameters)
		print("_q_function : \n", self._q_function)
		print("_epsilon : \n", self._epsilon)
		print("_gamma : \n", self._gamma)
		print("----------------------------------")
		return

	def get_q_functions(self):
		"""
		Q関数の値を取得する
		"""
		return self._q_function

	def delay_epsilon(self):
		"""
		ε-Greedy法のεを減衰する
		"""
		self._epsilon = self._epsilon / 1.10
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
		rand = np.random.rand()
		if self._epsilon >= rand:
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
		[a, b] = brain_parameters.shpae
		q_function = np.random.rand(a, b) * brain_parameters
		return q_function

	def update_q_function(self, s_a_r_historys):
		"""
		Brainを更新
		[Args]
			s_a_r_historys：<list> １エピソードでの(s, a, r)の履歴
			total_reward：<float> 割引利得の総和
		"""
		from collections import defaultdict
		N = defaultdict(lambda: [0] * self._n_actions)

		# 逐次訪問MC法による方策評価
		# t = 0 ~ T
		for (t, s_a_r) in enumerate(s_a_r_historys):
			state = s_a_r[0]
			action = s_a_r[1]
			reward = s_a_r[2]

			# i = t ~ T / 0 ~ T, 1 ~ T, 2 ~ T, ...
			k = 0
			for i in range(t, len(s_a_r_historys)):
				total_reward += self._gamma**k * s_a_r_historys[i][2]
				k += 1

			# 状態行動対のカウント
			N[state][action] += 1

			# 学習率(平均化のた逆数)
			alpha = 1 / N[state][action]

			# ゴール状態での行動価値観数は定義していないため,　除外　
			if state != 8:
				self._q_function[state][action] += alpha * (total_reward - self._q_function[state][action])

		return