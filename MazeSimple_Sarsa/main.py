import numpy as np
import random

# 自作モジュール
from Academy import Academy
from MazeAcademy import MazeAcademy

from Agent import Agent
from MazeAgent import MazeAgent

from Brain import Brain
from MazeSarsaBrain import MazeSarsaBrain

# 設定可能な変数
NUM_EPISODE = 100          # エピソード試行回数
NUM_TIME_STEP = 500        # 1エピソードの最大時間ステップ数
AGENT_NUM_STATES = 8       # 状態の要素数(s0 ~ s8)
AGENT_NUM_ACTIONS = 4      # 行動の要素数
AGENT_INIT_STATE = 0       # 初期状態の位置
BRAIN_LEARNING_RATE = 0.1  # 学習率
BRAIN_GREEDY_EPSILON = 0.5 # ε値
BRAIN_GAMMA = 0.99         # 割引率


def main():
	"""
	強化学習の学習環境用の迷路探索問題
	・エージェントの経路選択ロジックはSarsa
	"""
	print('Start main()')

	random.seed(1)
	np.random.seed(1)

	#==========================
	# 学習環境・エージジェント生成
	#==========================
	#--------------------------
	# Academy(envの役割)の生成
	#--------------------------
	academy = MazeAcademy(
		max_episode=NUM_EPISODE,
		max_time_step=NUM_TIME_STEP,
		save_step=25
	)

	#--------------------------
	# Brainの生成
	#--------------------------
	# 行動方策のためのパラメータを表形式(行：状態s、列：行動a)で定義
	# ＊行動方策を表形式で実装するために、対応するパラメータも表形式で実装する
	# 進行方向に壁があって進めない場合は、np.nanで初期化する
	# s8はゴール状態であるため、パラメータは定義しない
	# -----------
	# | s0 s1 s2|
	# |    --   |
	# | s3 s4|s5|
	# |       --|
	# | s6|s7 s8|
	# -----------
	brain_parameters = np.array(
		[	# a0='Up', a1='Right', a2='Down', a3='Left'
			[np.nan,      1,      1, np.nan], # s0
			[np.nan,      1, np.nan,      1], # s1
			[np.nan, np.nan,      1,      1], # s2
			[     1,      1,      1, np.nan], # s3
			[np.nan, np.nan,      1,      1], # s4
			[     1, np.nan, np.nan, np.nan], # s5
			[     1, np.nan, np.nan, np.nan], # s6
			[     1,      1, np.nan, np.nan], # s7			
		]
	)
	brain = MazeSarsaBrain(
		n_states=AGENT_NUM_STATES,
		n_actions=AGENT_NUM_ACTIONS,
		brain_parameters=brain_parameters,
		epsilon=BRAIN_GREEDY_EPSILON,
		gamma=BRAIN_GAMMA,
		learning_rate=BRAIN_LEARNING_RATE
	)
	
	#---------------------------
	# Agentの生成
	#---------------------------
	agent = MazeAgent(
		brain=brain,
		gamma=BRAIN_GAMMA,
		state0=AGENT_INIT_STATE
	)

	# AgentのBrainを設定
	agent.set_brain(brain)

	# 学習環境に作成したAgentを追加
	academy.add_agent(agent)

	agent.print('after init()')
	brain.print('after init()')

	#============================
	# エピソードの実行
	#============================
	academy.academy_run()
	agent.print('after simulation')
	brain.print('after simulation')

	return


if __name__=='__main__':
	main()
	
