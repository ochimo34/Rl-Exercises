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
		self._q_function_historys = []
		self._v_function_historys = []
		return

	def print(self, str):
		print("----------------------------------")
		print("MazeAgent")
		print(self)
		print(str)
		print("_brain : ", self._brain)
		print("_observations : ", self._observations)
		print("_total_reward : ", self._total_reward)
		print("_gamma : ", self._gamma)
		print("_done : ", self._done)
		print("_state : ", self._state)
		print("_action : ", self._action)
		print("_s_a_historys : ", self._s_a_historys)
		print("_reward_historys : ", self._reward_historys)
		print("----------------------------------" )
		return

	def get_q_function_historys(self):
		return self._q_function_historys

	def get_v_function_historys(self):
		return self._v_function_historys

	def collect_observations(self):
		"""
		Agentが観測している状態をBrainに提供する
		・Brainがエージェントの状態を取得時にコールバック
		"""
		return 

	def agent_reset(self):
		"""
		エージェントの再初期化処理
		"""
		self._done = False
		self._total_reward = 0.0

		# s0:エージェントの状態を再初期化して、開始位置に設定する
		# a0はまだわからないので、np.nan
		#__s_a_historys = [s0, np.nan]
		self._state = self._s_a_historys[0][0]
		self._action = self._s_a_historys[0][1]
		self._s_a_historys = [[self._state, self._action]]

		# a0：初期行動を設定
		self._action = next_action = self._brain.action(state=self._state)
		self._s_a_historys[-1][1] = self._action # -1で末尾に追加
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

		# 履歴の追加
		self._s_a_historys.append([next_state, next_action])

		# a' -> r''
		reward = 0.0
		if next_state == 8:
			reward = 1.0
		else:
			reward = -0.01
		self.add_reward(reward, time_step)

		# Q値の更新
		self._brain.update_q_function(state=self._state, action=self._action, next_state=next_state, next_action=next_action, reward=reward)

		# ゴールの指定
		self._state = next_state
		self._action = next_action
		if next_state == 8:
			self._done = True
		else:
			self._done = False

		return self._done

	def agent_on_done(self, episode, time_step):
		"""
		Academyのエピソード完了後にコールされる
		・Academyからコールされるコールバック関数
		"""
		#===================================
		# エピソード完了後の処理
		#===================================
		if episode % 10 == 0:
			print('エピソード = {0} / 最終時間ステップ数 = {1}'.format(episode, time_step))
			print('迷路を解くのにかかったステップ数:' + str(len(self._s_a_historys)))

		# εの衰退
		self._brain.decay_epsilon()

		# 累積報酬の追加
		self._reward_historys.append(self._total_reward)

		#------------------------------------
		# Q関数とV関数
		#------------------------------------
		q_function = self._brain.get_q_function()

		# エピソード開始時点との差分
		if episode == 0:
			# 初回エピソードの場合は履歴がないので、初期値
			# deep copyしたものをappend
			copy_q_function = self._brain.get_q_function().copy()
			self._q_function_historys.append(copy_q_function)
			copy_v_function = np.nanmax(copy_q_function, axis=1)
			self._v_function_historys.append(copy_v_function)

		# 状態価値の算出
		new_v_function = np.nanmax(q_function, axis=1)
		v_function = np.nanmax(self._q_function_historys[-1], axis=1)
		delta_v_function = np.sum(np.abs(new_v_function, v_function))
		if episode % 10 == 0:
			print('V 関数の大きさ：', np.abs(new_v_function))
			print('前回のエピソードのV関数との差分:', delta_v_function)

		self._q_function_historys.append(q_function.copy())
		self._v_function_historys.append(new_v_function)

		return