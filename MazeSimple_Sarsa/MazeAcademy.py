import numpy as np

# 自作クラス
from Academy import Academy
from Agent import Agent


class MazeAcademy(Academy):
	"""
	迷宮問題のエージェントの学習環境

	[public]

	[protected]

	[private]

	"""
	def __init__(self, max_episode=1, max_time_step=100, save_step=100):
		super().__init__(max_episode, max_time_step, save_step) # Academyクラスの引き継ぎメソッド
		return