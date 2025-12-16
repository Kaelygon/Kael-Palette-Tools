import time

class RorrLCG:
	"""
		Python random with seed is inconsistent and we only need randomish numbers
		Simple rotate right LCG for deterministic results
	"""  
 
	#because of rorr, lcg must be (mul%4==1 AND add%2==1) OR (mul%4==3 AND add%2==0)
	#Though full period isn't guaranteed #No cycle in first 1 billion with starting seed 0 
	MUL=13238717
	ADD=11770513
 
	LCG_BITS=64
	LCG_MOD=2**LCG_BITS-1
	LCG_SHIFT=int((LCG_BITS+1)/3)
	LCG_SHIFT_INV=LCG_BITS-LCG_SHIFT
 
	def __init__(self, in_seed=0):
		self._randInt: int = 0
		self.seed(in_seed)

	def _rorrlcg(self, num):
		num=(num>>self.LCG_SHIFT)|((num<<self.LCG_SHIFT_INV)&self.LCG_MOD) #RORR
		num=(num*self.MUL+self.ADD)&self.LCG_MOD #LCG, 
		return num

	#unsigned int
	def ui(self):
		self._randInt=self._rorrlcg(self._randInt)
		return self._randInt

	#float
	def f(self):
		return self.ui()/self.LCG_MOD

	def seed(self, in_seed: int = None):
		if in_seed == 0 or in_seed == None:
			self._start_seed = self.hash(str(time.time_ns()))
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

	def hash(self, string): 
		out_num=4611686018427384821 #null string value, any will do
		for c in string:
			out_num+= ord(c)
			out_num&= self.LCG_MOD
			out_num = self._rorrlcg(out_num)
		return out_num