#OkTools.py
from oklabConversion import *

import numpy as np


#Manipulate arrays of colors
class OkTools:
	FALLBACK_NORM = np.array([0.57735026918962576451]*3) * [1,-1,1] #sqrt(1/3)

	@staticmethod
	def manhattanDistance(vec, axis=1, keepdims=False):
		"""float[...] manhattanDistance(float[...] vec, int axis=1, bool keepdims=false)"""
		l_ratio = 0.66666666666666666667 # dist_exact/sum(abs(x,y,z)) = ~2/3 in 3D
		return np.sum(np.abs(vec), keepdims=keepdims, axis=axis) * l_ratio

	@staticmethod
	def vec3_arrayNorm(vector_list, axis=1):
		"""float[...] vec3_arrayNorm(float[...] vector_list, int axis=1, bool keepdims=false)"""
		l = np.linalg.norm(vector_list, axis=axis, keepdims=True)
		norm = vector_list / (l + 1e-12)
		norm[np.all(vector_list == 0, axis=axis)] = OkTools.FALLBACK_NORM
		return norm, np.squeeze(l, axis=-1)

	@staticmethod
	def vec3_norm(vector):
		"""float[3] vec3_norm(float[3] vector))"""
		l = np.linalg.norm(vector)
		if l==0:
			return OkTools.FALLBACK_NORM
		return vector/l

	@staticmethod
	def inOklabGamut(lab_list, eps = 1e-12):
		"""bool[] inOklabGamut(float[][3] lab_list, float eps = 1e-12))"""
		lin_list = oklabToLinear(lab_list)
		in_gamut = (lin_list >= -eps) & (lin_list <= 1+eps)
		in_gamut = in_gamut.all(axis=1)
		return in_gamut

	@staticmethod
	def clipToOklabGamut(lab_list, eps = 1e-12):
		"""(float[][3] float[][3]) clipToOklabGamut(float[][3] lab_list, float eps = 1e-12))"""
		lin_list = oklabToLinear(lab_list)
		out_gamut = (lin_list < -eps) | (lin_list > 1+eps)
		out_gamut = out_gamut.any(axis=1)

		if not np.any(out_gamut):
			return lab_list, None

		new_pos = np.clip(lin_list[out_gamut],[0.0]*3,[1.0]*3)
		new_lab = linearToOklab(new_pos)

		clip_move = np.zeros_like(lab_list)
		clip_move[out_gamut] = new_lab - lab_list[out_gamut] #movement in ok space

		out_list = lab_list.copy()
		out_list[out_gamut] = new_lab
		return out_list, clip_move

	@staticmethod
	def calcChroma(lab_list):
		"""float[] calcChroma(float[][3] lab_list))"""
		return np.sqrt( lab_list[:,1]**2 + lab_list[:,2]**2 )

	@staticmethod
	def isOkSrgbGray(lab_list, threshold = 1.0/255.0):
		"""bool[] isOkSrgbGray(float[][3] lab_list, float threshold = 1.0/255.0))"""
		rgb_list = oklabToSrgb(lab_list)
		is_gray = (
			(abs(rgb_list[:,0]-rgb_list[:,1]) < threshold) & 
			(abs(rgb_list[:,1]-rgb_list[:,2]) < threshold)
		)
		return is_gray

	@staticmethod
	def srgbToHex(rgb):
		"""char* srgbToHex(float[3] rgb)"""
		rgb = np.clip(rgb,[0.0]*3,[1.0]*3)
		rgb = np.round(rgb * 255.0)	
		rgb = rgb.astype(np.uint8)
		return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])
