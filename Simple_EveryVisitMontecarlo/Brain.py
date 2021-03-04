class Brain(object):
	"""
	エージェントの意思決定ロジック
	・複数のエージェントが同じ意思決定ロジックを共有できるように、Brainとしてclass化する。
	・移動方法などのActionを設定する。

	[public]

	[protected] 変数名の前にアンダースコア(_)をつける
		_n_states：<int> 状態の要素数
		_n_actions：<int> 行動の要素数

	[private] 変数名の前にダブルアンダースコア(__)をつける

	"""
	def __init__(self, n_states, n_actions):
		self._n_states = n_states
		self._n_actions = n_actions
		return

	def print(self, str):
		print("----------------------------------")
		print("Brain")
		print(self)
		print(str)
		print("_n_states : \n", self._n_states)
		print("_n_actions : \n", self._n_actions)
		print("----------------------------------")
		return

	def reset_brain(self):
		"""
		Brainを再初期化する
		"""
		self._policy = None
		return
		