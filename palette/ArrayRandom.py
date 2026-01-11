import numpy as np
import time

#https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
class ArrayRandom:
	"""
		Splitmix64 vectorized pseudo number generator
		If no seed is given, a random is generated
	"""
	_state = None
	_seed = None #ArrayRandom._seed can be passed to new instance to reproduce the results
	MASK = 2**64-1

	FLOAT_MASK = np.uint64(1023) << 52
	CONST_ADD1 = np.uint64(0x9e3779b97f4a7c15)
	CONST_MUL1 = np.uint64(0xbf58476d1ce4e5b9)
	CONST_MUL2 = np.uint64(0x94d049bb133111eb)

	#Python int or np.int
	def __init__(self, user_seed = None):
		buf_seed = user_seed
		if buf_seed is None:
			#effective starting seed is the output of 1 iteration pass, using _state = time + CONST_ADD1
			self._state = np.uint64(time.perf_counter_ns())
			buf_seed = self.randomInt((1,))[0]
		elif not isinstance(buf_seed, (int,np.uint64)):
			raise ValueError("user_seed must be int or np.uint64")
		self._state = np.uint64(buf_seed & self.MASK)
		self._seed = self._state

	def randomInt(self, shape: tuple):
		"""
			No safety checks. Shape must be tuple (>=1,) and its product mustn't be higher than 2**63-1
			but you'll run out of memory long before that.
		"""
		#have to use np.<operand>() to implicitly suppress overflow warnings
		count = np.prod(shape, dtype=int)

		rand_arr = np.arange(1, count+1, dtype=np.uint64) #n=1,2,3...
		rand_arr*=self.CONST_ADD1 #i*n
		rand_arr+=self._state #i*n+s

		#Splitmix64
		rand_arr^= rand_arr >> 30
		rand_arr*= self.CONST_MUL1
		rand_arr^= rand_arr >> 27
		rand_arr*= self.CONST_MUL2
		rand_arr^= rand_arr >> 31

		self._state = np.add(self._state, np.multiply(self.CONST_ADD1, np.uint64(count)) )
		return rand_arr.reshape(shape)

	def random(self, shape: tuple):
		"""randomInt(shape: tuple) -> normalized [0,1) """
		return (self.randomInt(shape) >> 12  | self.FLOAT_MASK).view(np.float64) - 1.0

	def shuffle(self, arr):
		count = arr.shape[0]
		rand_arr = self.randomInt((count,))
		order = np.argsort(rand_arr, kind='stable')
		arr[:] = arr[order]