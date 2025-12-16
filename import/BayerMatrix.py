import numpy as np

"""
Ordered matrix creation tools
"""

#Normilze to (-0.5,0.5)
def Ordered_normalize(b_m):
	if np.all(b_m == b_m.flat[0]):
		return np.zeros_like(b_m)
	b_h, b_w = b_m.shape
	b_m -= np.min(b_m) #[max,min] -> [max-min,0]
	b_m = b_m / np.max(b_m) # -> [1,0]
	b_m-= 0.5 # -> [0.5,-0.5]
	b_m = b_m*(1-(2e-6)) #theshold extremes to preserve e.g. pure black/white
	return b_m

#Return n*n*m bayer matrix. Non powers of 2 are bayer-like. n<1 is invalid 
def Bayer_createMatrix(n: int, m: int=1):
	if n<1:
		#fallback
		amogus = (343150 >> np.arange(25) & 1).reshape(5,5)
		return np.stack([amogus]*m, axis=2)

	x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
	b_m = np.zeros_like(x)
	s = 1
	mapping = np.array([0, 6, 2, 4, 3, 5, 1, 7]) #https://jbaker.graphics/writings/bayer.html

	while s < n:
		index = (((x // s) % 2) << 2) | (((y // s) % 2) << 1) | ((z // s) % 2)
		b_m = b_m * 8 + mapping[index]
		s <<= 1

	#choose first m slices, pad with last slice (that way if n=2, oklab chroma a,b shares same slice)
	chosen_slices = np.clip(np.arange(m), 0, n-1)
	b_m_slices = b_m[:, :, chosen_slices]
 
	arr_ranks = np.empty_like(b_m_slices)
	for c in range(m):
		slice_flat = b_m_slices[:, :, c].ravel()
		order = np.argsort(slice_flat)
		ranks = np.empty_like(slice_flat)
		ranks[order] = np.arange(0, slice_flat.size)
		arr_ranks[:, :, c] = ranks.reshape(n, n)

	return arr_ranks

#Take first 3 slices of 3D bayer matrix with side length of n, normalized to -0.5,0.5
def Ordered_bayerOkLab(n):
	b_m = Bayer_createMatrix(n, 3)
  
	m_l = b_m[:,:,0]
	m_a = b_m[:,:,1]
	m_b = b_m[:,:,2]
	m_l = Ordered_normalize(m_l)
	m_a = Ordered_normalize(m_a)
	m_b = Ordered_normalize(m_b)
	return m_l, m_a, m_b
