import numpy as np


class Agent(object):
	"""
	強化学習におけるエージェントをモデル化したクラス.
	・実際のAgentクラスの実装は、このクラスを継承し、オーバライドを想定。

	[public]

	[protected] 変数名の前にアンダースコア(_)をつける
		_brain：<Brain> エージェントのBrainへの参照
		_observations：list<動的な型> エージェントが観測できる状態
		_total_reward：<float> 割引報酬の総和
		_gamma：<float> 割引率
		_done：<bool> エピソード完了フラグ
		_state：<int> エージェントの現在の状態 s
		_action：<int> エピソードの現在の行動 a
		_s_a_historys：list<[int, int]> エピソードの状態と行動の履歴
		_reward_historys：list<float> 割引報酬の履歴/shape=[n_episode]

	[private] 変数名の前にダブルアンダースコア(__)をつける

	"""
	def __init__(self, brain=None, gamma=0.9, state0=0):
		self._brain = brain
		self._observations = []
		self._total_reward = 0.0
		self._gamma = gamma
		self._done = False
		self._state = state0
		self._action = np.nan
		self._s_a_historys = [[self._state, self._action]]
		self._reward_historys = [self._total_reward]
		return

	def print(self, str):
		print("----------------------------------")
		print("Agent")
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
		print("----------------------------------")
		return

	def get_s_a_historys(self):
		return self._s_a_historys

	def get_reward_historys(self):
		return self._reward_historys

	def collect_observations(self):
		"""
		Agentが観測しているstateをbrainに提供する。
		・Brainがエージェントの状態を取得時にコールバック(呼び出す)する。
		"""
		self._observations = []
		return self._observations

	def set_brain(self, brain):
		"""
		エージェントのBrainを設定する。
		"""
		self._brain = brain
		return

	def add_vector_obs(self, observation):
		"""
		エージェントが観測できる状態を追加する。
		"""
		self._observations.append(observation)
		return

	def done(self):
		"""
		エピソードを完了にする。
		"""
		self._done = True
		return

	def is_done(self):
		"""
		Academyがエピソードを完了したかの取得
		"""
		return self._done

	def set_total_reward(self, total_reward):
		"""
		報酬をセットする
		"""
		self._total_reward = total_reward
		return self._total_reward

	def add_reward(self, reward, time_step):
		"""
		報酬を加算する。
		・割引収益 Rt = Σ_t{γ^t*r_(t+1)}
		"""
		self._total_reward += (self._gamma**time_step)*reward
		return self._total_reward

	def agent_reset(self):
		"""
		エージェントの再初期化処理
		"""
		self._total_reward = 0.0
		self._done = False
		self._state = self._s_a_historys[0][0]
		self._action = self._s_a_historys[0][1]
		self._s_a_historys = [[self._state, self._action]]
		return

	def agent_step(self, episode, time_step):
		"""
		エージェントの次の状態を決定する。
		・Academyから各時間ステップごとにコールされるコールバック関数。

		[Args]
			episode：<int>現在のエピソード数
			time_step：<int>現在の時間ステップ

		[Returns]
			done：<bool>エピソードの完了フラグ
		"""
		self._done = False
		return self._done

	def agent_on_done(self, episode, time_step):
		"""
		Academyのエピソード完了後にコールされ、エピソードの終了時の処理を記述する。
		・Academyからコールされるコールバック関数

		[args]
			episode:<int>現在のエピソード数
			time_step:<int>エピソード完了時の時間ステップ数
		"""
		return
