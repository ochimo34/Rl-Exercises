import numpy as np
import matplotlib.pyplot as plt


# 自作クラス
from Agent import Agent


class Academy(object):
	"""
	エージェントの強化学習環境
	・強化学習モデルにおける環境Environmentに対応
	・学習や推論を行うための設定を行う

	[public]

	[protected] 変数名の前にアンダースコア(_)をつける
		_max_episode：<int> エピソードの最大回数。最大回数に到達すると、Academyと全Agentのエピソードを完了する。
		_max_time_step：<int> 時間ステップの最大回数
		_save_step：<int> 保存間隔(エピソード数)
		_agents：list<AgentBase>
		_done：<bool> エピソードが完了したかのフラグ

	[private] 変数名の前にダブルアンダースコア(__)をつける

	"""
	def __init__(self, max_episode=1, max_time_step=100, save_step=5):
		self._max_episode = max_episode
		self._max_time_step = max_time_step
		self._save_step = save_step
		self._agents = []
		self._done = False
		return

	def academy_reset(self):
		"""
		学習環境をリセットする
		"""
		if (self._agents != None):
			for agent in self._agents:
				agent.agent_reset()

		self._done = False
		return
	
	def done(self):
		"""
		エピソードを完了にする
		"""
		self._done = True
		return

	def is_done(self):
		"""
		Academyがエピソードを完了したかの取得
		"""
		return self._done

	def add_agent(self, agent):
		"""
		学習環境にエージェントを追加する。
		"""
		self._agents.append(agent)
		return

	def academy_run(self):
		"""
		学習環境を実行する
		"""
		# エピソードを施行
		for episode in range(self._max_episode):
			print('エピソード：', episode)

			# 学習環境をリセット
			self.academy_reset()

			# 時間ステップを1ステップずつ進める
			for time_step in range(self._max_time_step):
				dones = []

				for agent in self._agents:
					done = agent.agent_step(episode, time_step)
					dones.append(done)

				# 全エージェントが完了した場合
				if all(dones) == True:
					break

			# Academyと全Agentsのエピソードを完了
			self._done = True
			for agent in self._agents:
				agent.agent_on_done(episode, time_step)

		return

	def add_frame(self, episode, time_step):
		"""
		強化学習の環境の1フレームをリストに追加する
		"""
		# frame = None
		# self._frames.append(frame)
		return

	def save_frames(self, file_name):
		"""
		外部ファイルに動画を保存する
		"""
		return