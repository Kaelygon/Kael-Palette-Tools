import numpy as np

"""
Bayer matrix creation tools
I don't even pretend like I know how this works but LGTM
"""

def _Bayer_calcCorrelationMap(p, q, w):
	# Corr_ = ifft2( fft2(w*p) * conj(fft2(q)) )
	Fp = np.fft.fft2(p)
	Fq = np.fft.fft2(w * q)
	corr = np.fft.ifft2(Fp * np.conj(Fq))
	return np.real(corr)

#calc layered bayer matrix shifts such that they minimize interference
def _Bayer_calcShifts(m_l, m_a, m_b, w):
	H, W = w.shape

	Corr_la = _Bayer_calcCorrelationMap(m_l, m_a, w)  # Corr_{L,A}(s) = sum w * L(x+s)*A(x)
	Corr_lb = _Bayer_calcCorrelationMap(m_l, m_b, w)
	Corr_ab = _Bayer_calcCorrelationMap(m_a, m_b, w)

	best = ((0,0),(0,0),(0,0))
	best_obj = np.inf
	#Shifts are relative to s_L = (0,0)
	for ay in range(H):
		for ax in range(W):
			idx_la = ((-ay) % H, (-ax) % W)
			val_la = Corr_la[idx_la]

			for by in range(H):
				for bx in range(W):
					idx_lb = ((-by) % H, (-bx) % W)
					idx_ab = ((ay - by) % H, (ax - bx) % W)

					val_lb = Corr_lb[idx_lb]
					val_ab = Corr_ab[idx_ab]

					obj = val_la + val_lb + val_ab

					if obj < best_obj:
						best_obj = obj
						best = ((0,0), (ay, ax), (by, bx))
	return best

def _Bayer_calcTileWeights(pixel_gaps, y_idxs, x_idxs, b_h, b_w):
	 w = np.zeros((b_h, b_w), dtype=np.float64)
	 np.add.at(w, (y_idxs % b_h, x_idxs % b_w), pixel_gaps)
	 return w


#Return NxN bayer matrix. Non powers of 2 are bayer-like. n<1 is invalid 
def Bayer_createMatrix(n: int):
	if n<1:
		return (343150 >> np.arange(25) & 1).reshape(5,5) #easter egg

	X, Y = np.meshgrid(np.arange(n), np.arange(n))
	M = np.zeros_like(X)
	s = 1
	mapping = np.array([0, 3, 2, 1])
	while s < n:
		pair = ((X // s) % 2) << 1 | ((Y // s) % 2)
		M = M * 4 + mapping[pair]
		s <<= 1
	return M

#return bayer thresholds for each pixel. 
#pixel_gaps is list of distance vectors of two nearest palette colors of every pixel np.array[[float]*3]
#calculated from three staggered Bayer matrices of same size that minimize interference when layered
def Bayer_calcPixelThresholds(matrix_size, pixel_gaps, image_width):

	#bayer matrix thresholds
	b_m=Bayer_createMatrix(int(matrix_size))
	b_h, b_w = b_m.shape
	max_cell = np.max(b_m)
	if max_cell == 0: #avoid div by 0
		b_m = b_m*0.0
	else:
		b_m = b_m / max_cell - 0.5 #normalize [-0.5,0.5]
	b_m = b_m*(1-(2e-6)) #theshold extremes (-0.5+eps,0.5-eps)

	y_idxs, x_idxs = np.divmod(np.arange(len(pixel_gaps)), image_width)

	#Stagger matrices to reduce channel interference
	m_l = b_m
	m_a = np.rot90(m_l)
	m_b = np.flipud(m_a)

	w = _Bayer_calcTileWeights(pixel_gaps, y_idxs, x_idxs, b_h, b_w)
	s_l, s_a, s_b = _Bayer_calcShifts(m_l, m_a, m_b, w)

	m_l = np.roll(m_l, shift=s_l, axis=(0,1))
	m_a = np.roll(m_a, shift=s_a, axis=(0,1))
	m_b = np.roll(m_b, shift=s_b, axis=(0,1))

	m_l = m_l[y_idxs % b_h, x_idxs % b_w]
	m_a = m_a[y_idxs % b_h, x_idxs % b_w]
	m_b = m_b[y_idxs % b_h, x_idxs % b_w]
	
	return m_l, m_a, m_b
