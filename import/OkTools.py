from oklabConversion import *

import numpy as np
from numpy.typing import NDArray


#Manipulate arrays of colors
class OkTools:
	FALLBACK_NORM = np.array([0.57735026918962576451]*3) * [1,-1,1] #sqrt(1/3)

	def vec3_array_norm(vector_list: NDArray[[float]*3], axis=1):
		l = np.linalg.norm(vector_list, axis=axis, keepdims=True)
		l[l==0] = 1.0
		norm = vector_list / l
		norm[np.all(vector_list == 0, axis=axis)] = OkTools.FALLBACK_NORM
		return norm

	def vec3_norm(vector: [float]*3):
		l = np.linalg.norm(vector)
		if l==0:
			return OkTools.FALLBACK_NORM
		return vector/l

	@staticmethod
	def inOklabGamut(lab_list, eps:float=1e-12):
		lin_list = oklabToLinear(lab_list)
		in_gamut = (lin_list >= -eps) & (lin_list <= 1+eps)
		in_gamut = in_gamut.all(axis=1)
		return in_gamut

	@staticmethod
	def clipToOklabGamut(lab_list, eps:float=1e-12):
		lin_list = oklabToLinear(lab_list)
		out_gamut = (lin_list < -eps) | (lin_list > 1+eps)
		out_gamut = out_gamut.any(axis=1)

		if np.any(out_gamut):
			new_pos = np.clip(lin_list[out_gamut],[0.0]*3,[1.0]*3)
			new_lab = linearToOklab(new_pos)
			clip_move = np.zeros_like(lab_list)
			clip_move[out_gamut] = new_lab - lab_list[out_gamut] #movement in ok space
			lab_list[out_gamut] = new_lab
			return lab_list, clip_move

		return lab_list, None

	@staticmethod
	def calcChroma(lab_list):
		return np.sqrt( lab_list[:,1]**2 + lab_list[:,2]**2 )

	@staticmethod
	def isOkSrgbGray(lab_list, threshold: float = 1.0/255.0):
		rgb_list = oklabToSrgb(lab_list)
		is_gray = (
			(abs(rgb_list[:,0]-rgb_list[:,1]) < threshold) & 
			(abs(rgb_list[:,1]-rgb_list[:,2]) < threshold)
		)
		return is_gray

	### misc tools

	@staticmethod
	def srgbToHex(rgb: [float]*3):
		rgb = np.clip(rgb,[0.0]*3,[1.0]*3)
		rgb = np.round(rgb * 255.0)	
		rgb = rgb.astype(np.uint8)
		return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])
