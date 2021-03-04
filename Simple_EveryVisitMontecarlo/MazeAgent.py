import numpy as np

# 自作クラス
from Agent import Agent


class MazeAgent(Agent):
	"""
	迷路探索用エージェント

	[public]

	[protected]

	[private]

	"""
	def __init__(self, brain=None, gamma=0.9, state0=0):
		super().__init__(brain, gamma, state0)
		self._s_a_r_historys = [[self._state, self._action, 0.0]]
		self._q_function_historys = []
		self._v_function_historys = []
		return

	def print(self, str):
		print("----------------------------------")
		print("MazeAgent")
		print(self)
		print(str)
		print("_brain : \n", self._brain)
		print("_observations : \n", self._observations)
		print("_total_reward : \n", self._total_reward)
		print("_gamma : \n", self._gamma)
		print("_done : \n", self._done)
		print("_state : \n", self._state)
		print("_action : \n", self._action)
		print("_s_a_r_historys : \n", self._s_a_r_historys)
		print("_reward_historys : \n", self._reward_historys)
		print("----------------------------------" )
		return

	def get_q_function_historys(self):
		return self._q_function_historys

	def get_v_function_historys(self):
		return self._v_function_historys

	def agent_reset(self):
		"""
		エージェントの再初期化処理
		"""
		self._done = False
		self._total_reward = 0.0

		# s0:エージェントの状態を再初期化して、開始位置に設定する
		# a0はまだわからないので、np.nan
		#__s_a_historys = [s0, np.nan]
		self._state = self._s_a_r_historys[0][0]
		self._action = self._s_a_r_historys[0][1]
		self._s_a_historys = [[self._state, self._action]]
		self._s_a_r_historys = [[self._state, self._action, self._total_reward]]

		# a0：初期行動を設定
		self._action = next_action = self._brain.action(state=self._state)
		self._s_a_historys[-1][1] = self._action # -1で末尾に追加
		self._s_a_r_historys[-1][1] = self._action
		return

	def agent_step(self, episode, time_step):
		"""
		エージェントの次の状態を決定
		・Academyから各時間ステップ毎にコールされる

		[Args]
			episode：現在のエピソード数
			time_step：現在の時間ステップ

		[Returns]
		done：<bool> エピソード完了フラグ
		"""
		# 既にエピソードが完了状態なら、そのままreturnして全エージェントの完了を待つ
		if self._done == True:
			return self._done

		#----------------------------------------------------------------------
		# １時間ステップでの迷宮探索
		# バックアップ線図 : s_t → a_t → r_(t+1) → s_(t+1) → a_(t+1) → r_(t+2) → Q → ...
		#----------------------------------------------------------------------
		# r → a → s'：行動aに従った次状態s'の決定
		if self._action == 0: # Up
			next_state = self._state - 3
		elif self._action == 1: # Right
			next_state = self._state + 1
		elif self._action == 2: # Down
			next_state = self._state + 3
		elif self._action == 3: # Left
			next_state = self._state - 1
		else: # np.nanなど
			next_state = self._state

		# s' → a'：次状態s'での次行動a'
		if next_state == 8:
			# next_action = np.nan
			next_action = 0
		else:
			next_action = self._brain.action(state=next_state)

		# Brainの更新処理
		# self._brain.update_q_function()
		


























