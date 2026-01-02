"""
Obsolete, since we are switching from python.rand to numpy.random
"""

import time
import math

class RorrLCG:
	"""
		Python random with seed is inconsistent and we only need randomish numbers
		Simple rotate right LCG for deterministic results
	"""

	#because of rorr, lcg must be (mul%4==1 AND add%2==1) OR (mul%4==3 AND add%2==0)
	MUL=377317001288688793
	ADD=229269304956507589

	LCG_BITS=64 #must be power of 2 and divisible by 8
	LCG_MASK=2**LCG_BITS-1
	LCG_SHIFT=int((LCG_BITS+1)/3)
	LCG_SHIFT_INV=LCG_BITS-LCG_SHIFT

	def __init__(self, in_seed=0):
		self._randInt: int = 0
		self.seed(in_seed)

	def _rorrlcg(self, num):
		num=(num>>self.LCG_SHIFT)|((num<<self.LCG_SHIFT_INV)&self.LCG_MASK) #RORR
		num=(num*self.MUL+self.ADD)&self.LCG_MASK #LCG,
		return num

	#unsigned int
	def ui(self):
		self._randInt=self._rorrlcg(self._randInt)
		return self._randInt

	def f(self):
		return (self.ui()+1)/(self.LCG_MASK+2)

	def seed(self, in_seed: int = None):
		if in_seed == 0 or in_seed == None:
			self._start_seed = self.hash(str(time.time_ns()))
		elif in_seed:
			self._start_seed = in_seed&self.LCG_MASK
		self._randInt=self._start_seed
		self.ui()
		print("Seed: "+str(self._start_seed)+"\n")

	def shuffleDict(self, data):
		items = list(data.items())
		list_size = len(items)
		for i in range(list_size - 1, 0, -1):
			j = self.ui() % (i + 1)
			items[i], items[j] = items[j], items[i]
		return dict(items)

	def uniform_f(self, _min, _max):
		rand_f = self.f()
		return rand_f*(_max-_min) + _min

	def vec3(self):
		vec = (self.f()-0.5, self.f()-0.5, self.f()-0.5)
		l = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
		return (vec[0]/l, vec[1]/l, vec[2]/l)

	def hash(self, string, shuffle_count=6):
		_bytes = list(string.encode('utf-8'))

		out_num=4611686018427384821 #null string value, can be any
		for i,c in enumerate(_bytes):
			shift = (i*8) & (self.LCG_BITS-1)
			out_num ^= self._rorrlcg(c<<shift)

		#shuffle if order matters
		for i in range(shuffle_count):
			out_num = self._rorrlcg(out_num)
		return out_num