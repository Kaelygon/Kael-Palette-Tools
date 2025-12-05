import time

class RorrLCG:
	"""
		Python random with seed is inconsistent and we only need randomish numbers
		Simple rotate right LCG for deterministic results
	"""  
	LCG_MOD=0xFFFFFFFFFFFFFFFF #2^64-1
	LCG_BITS=64
	LCG_SHIFT=int((LCG_BITS+1)/3)
	LCG_SHIFT_INV=LCG_BITS-LCG_SHIFT
	def __init__(self, in_seed=0):
		self._randInt: int = 0
		self.seed(in_seed)
	#unsigned int
	def ui(self):
		self._randInt=(self._randInt>>self.LCG_SHIFT)|((self._randInt<<self.LCG_SHIFT_INV)&self.LCG_MOD) #RORR
		self._randInt=(self._randInt*3343+11770513)&self.LCG_MOD #LCG
		return self._randInt

	#float
	def f(self):
		return self.ui()/self.LCG_MOD

	def seed(self, in_seed: int = None):
		if in_seed == 0 or in_seed == None:
			self._start_seed = hash(time.time_ns()) & self.LCG_MOD
		elif in_seed:
			self._start_seed = in_seed&self.LCG_MOD
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
		l = 0
		while l==0:
			vec = float3(self.f()-0.5, self.f()-0.5, self.f()-0.5)
			l = length_vec3(vec)
		return float3(vec[0]/l, vec[1]/l, vec[2]/l)