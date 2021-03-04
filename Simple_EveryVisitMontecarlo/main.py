import numpy as np


# 設定可能な変数
NUM_EPISODE = 100    # エピソード試行回数
NUM_TIME_STEP = 500  # 1エピソードの最大時間ステップ数


def main():
	"""
	強化学習の学習環境用の迷路探索問題
	・エージェントの経路選択ロジックは、逐一訪問モンテカルロ法
	"""
	print('Start main()')

	random.seed(0)
	np.random.seed(0)

	#==========================
	# 学習環境・エージジェント生成
	#==========================
	#--------------------------
	# Academy(envの役割)の生成
	#--------------------------
	academy = MazeAcademy(max_episode=NUM_EPISODE, max_time_step=NUM_TIME_STEP, save_step=25)
	