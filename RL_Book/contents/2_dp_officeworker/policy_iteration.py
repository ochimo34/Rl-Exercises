# モジュールのインポート
import numpy as np
import copy


# 定義
P = [0.8, 0.5, 1.0]     # MDP
GAMMA = 0.95            # 割引率
R = np.zeros((3, 2, 3)) # 報酬(s, a, s')
R[0, 0, 1] =  1.0
R[0, 0, 2] =  2.0
R[1, 0, 0] =  1.0
R[1, 0, 2] =  2.0
R[1, 1, 1] =  1.0
R[2, 0, 0] =  1.0
R[2, 1, 2] = -1.0


# 方策評価関数の定義
def policy_estimator(policy):
	# 初期化
	r = [0, 0, 0]
	p = np.zeros((3, 3))
	A = np.zeros((3, 3))

	for i in range(3):
		# 状態遷移行列の計算
		p[i, (i + 1) % 3] = policy[i] * P[i]       # 行動0
		p[i, (i + 2) % 3] = policy[i] * (1 - P[i]) # 行動0
		p[i, i] = 1 - policy[i] # 行動1を取ったとき同じ地点へ遷移

		# 報酬ベクトルの計算
		r[i] = policy[i] * (P[i] * R[i, 0, (i + 1) % 3] \
				+ (1 - P[i]) * R[i, 0, (i + 2) % 3]) \
				+ (1 - policy[i]) * R[i, 1, i]

	# 行列計算によるベルマン方程式の求解
	A = np.eye(3) - GAMMA * p
	B = np.linalg.inv(A)
	v_sol = np.dot(B, r)

	return v_sol


# 方策反復法による計算
def policy_iteration():
	# 初期化
	v = [0, 0, 0]            # 状態価値関数
	v_prev = copy.copy(v)
	q = np.zeros((3, 2))     # 行動価値関数
	policy = [0.5, 0.5, 0.5] # 方策(行動0を取る確率)

	for step in range(100):
		# 方策評価
		v = policy_estimator(policy)

		# 価値観数vが収束するまで
		if np.min(v - v_prev) <= 0:
			break

		# 現在の価値関数と方策を表示
		print('step:{}, value:{}, policy:{}'.format(step, v, policy))

		# 方策改善 
		for i in range(3):
			# Qを計算
			q[i, 0] = P[i] * (R[i, 0, (i + 1) % 3] + GAMMA * v[(i + 1) % 3]) \
						+ (1 - P[i]) * (R[i, 0, (1 + 2) % 3] + GAMMA * v[(i + 2) % 3])
			q[i, 1] = R[i, 1, i] + GAMMA * v[i]

			if q[i, 0] > q[i, 1]:
				policy[i] = 1
			elif q[i, 0] == q[i, 1]:
				policy[i] = 0.5
			else:
				policy[i] = 0


		# 価値観数の保存
		v_prev = copy.copy(v)
	return


# 実行
if __name__ == '__main__':
	policy_iteration()