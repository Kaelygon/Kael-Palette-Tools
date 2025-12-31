import numpy as np
from scipy.spatial import cKDTree as KDTree

"""
ordered. namespace matrix creation tools
"""
class ordered:
	#Normalize 3D matrix slices to (-0.5,0.5)
	@staticmethod
	def _normSlices(matrix):
		matrix = matrix.astype(float)
		for s in range(matrix.shape[2]):
			slice = matrix[:,:,s]
			slice = slice - np.min(slice) #[max,min] -> [max-min,0]
			if not slice.any():
				matrix[:,:,s] = np.zeros_like(slice)
				continue
			slice = slice / np.max(slice) # -> [1,0]
			slice-= 0.5 # -> [0.5,-0.5]
			matrix[:,:,s] = slice*(1-(2e-6)) #theshold extremes to preserve e.g. pure black/white
		return matrix


	#Order 3D matrix such that it contains 0 to x*y*z integers with no gaps or duplicates
	@staticmethod
	def _rankMatrix(matrix, height, width, depth):
		slice_flat = matrix.ravel()
		order = np.argsort(slice_flat)
		slice_flat[order] = np.arange(height*width*depth)
		arr_ranks = slice_flat.reshape(height,width,depth)

		return arr_ranks

	@staticmethod
	def _fallbackMatrix(depth):
		amogus = (343150 >> np.arange(25) & 1).reshape(5,5)
		return np.stack([amogus]*depth, axis=2)

	#Bayer
	#Take first 3 slices of 3D bayer matrix with side length of n, normalized to -0.5,0.5
	@staticmethod
	def bayerOklab(size, depth=3):
		if size<1:
			return ordered._fallbackMatrix(depth)

		x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
		b_m = np.zeros_like(x)
		s = 1
		mapping = np.array([0, 6, 2, 4, 3, 5, 1, 7]) #https://jbaker.graphics/writings/bayer.html

		while s < size:
			index = (((x // s) % 2) << 2) | (((y // s) % 2) << 1) | ((z // s) % 2)
			b_m = b_m * 8 + mapping[index]
			s <<= 1

		#choose first m slices, pad with last slice (that way if n=2, oklab chroma a,b shares same slice)
		chosen_slices = np.clip(np.arange(depth), 0, size-1)
		b_m_slices = b_m[:, :, chosen_slices]

		b_m = ordered._rankMatrix(b_m_slices,size,size,depth)
		b_m = ordered._normSlices(b_m)
	
		m_l = b_m[:,:,0]
		m_a = b_m[:,:,1]
		m_b = b_m[:,:,2]
		return m_l, m_a, m_b


	#Blue noise
	#lerp (white noise) -> (1 - avg neighborhood)
	#weight=0 is white noise, weight=1 is neighborhood avg
	def blueNoiseOklab(height, width, weight=0.64, channel_count = 3):
		if height<1 or width<1:
			return ordered._fallbackMatrix(channel_count)

		pixels = np.random.rand(height,width,channel_count)

		#pixels relative to current
		top	= np.roll(pixels, shift=(-1, 0, 0), axis=(0,1,2))
		bot	= np.roll(pixels, shift=( 1, 0, 0), axis=(0,1,2))
		left	= np.roll(pixels, shift=( 0,-1, 0), axis=(0,1,2))
		right	= np.roll(pixels, shift=( 0, 1, 0), axis=(0,1,2))
		front	= np.roll(pixels, shift=( 0, 0,-1), axis=(0,1,2))
		back	= np.roll(pixels, shift=( 0, 0, 1), axis=(0,1,2))
		
		neighborhood = np.stack([top, bot, left, right, front, back]) #(6,h,w,3)
		neighborhood_avg = np.average(neighborhood, axis=0) #-> (h,w,3)

		avg_invert = 1.0-neighborhood_avg
		new_pos = pixels*(1.0-weight) + avg_invert*(weight)
		pixels = ordered._normSlices(new_pos)

		pixels = ordered._rankMatrix(pixels,height,width,channel_count)
		pixels = ordered._normSlices(pixels)
	
		m_l = pixels[:,:,0]
		m_a = pixels[:,:,1]
		m_b = pixels[:,:,2]
		return m_l, m_a, m_b
