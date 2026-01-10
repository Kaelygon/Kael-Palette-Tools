import numpy as np
import time

class ArrayRandom:
	state = np.uint64(1)
	STATE_LENGTH = 2**64
	MASK = STATE_LENGTH-1

	def __init__(self, seed:np.uint64 = None): 
		self.state = seed
		if self.state is None:	
			#Random seed from time
			time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
			self.state = np.uint64(time_ns)
			self.state = self.randomInt(1)[0]
		self.state = np.uint64(self.state & self.MASK)

	def randomInt(self, shape):
		count = np.prod(shape)
		step = self.MASK//int(count)
		rand_arr = (np.arange(0, count, dtype=np.uint64)) * step + 1 + self.state

		for _ in range(2): #xorshift64star
			rand_arr ^= rand_arr >> 12
			rand_arr ^= rand_arr << 25
			rand_arr ^= rand_arr >> 27
			rand_arr = rand_arr * 0x2545F4914F6CDD1D
		
		self.state = rand_arr[0]
		rand_arr = rand_arr.reshape(shape)
		return rand_arr

	def random(self, shape):
		#normalized [0,1)
		return self.randomInt(shape) / float(self.STATE_LENGTH)

	def shuffle(self, arr):
		count = arr.shape[0]
		rand_arr = self.randomInt(count)
		order = np.argsort(rand_arr)
		arr[:] = arr[order]