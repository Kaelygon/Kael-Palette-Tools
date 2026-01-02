#OkTools.py
from oklabConversion import *

import numpy as np
from numpy.typing import NDArray


#Manipulate arrays of colors
class OkTools:

	def vec3_array_norm(vector_list: NDArray[[float]*3]):
		l = np.linalg.norm(vector_list,axis=1)
		non_zero_move = l!=0

		norm = np.zeros_like(vector_list) + [1,0,0]
		norm[non_zero_move] = vector_list[non_zero_move]/(l[non_zero_move][:,None])
		return norm

	def vec3_norm(vector: [float]*3):
		l = np.linalg.norm(vector)
		if l==0:
			return [1,0,0]
		return vector/l

	@staticmethod
	def inOklabGamut(lab_list, eps:float=1e-7):
		lin_list = oklabToLinear(lab_list)
		in_gamut = (lin_list >= -eps) & (lin_list <= 1+eps)
		in_gamut = in_gamut.all(axis=1)
		return in_gamut

	@staticmethod
	def clipToOklabGamut(lab_list, eps:float=1e-7):
		lin_list = oklabToLinear(lab_list)
		out_gamut = (lin_list < -eps) | (lin_list > 1+eps)
		out_gamut = out_gamut.any(axis=1)

		clip_move = np.zeros_like(lab_list)
	
		new_pos = np.clip(lin_list[out_gamut],[0.0]*3,[1.0]*3)
		if np.any(new_pos):
			new_lab = linearToOklab(new_pos)
			clip_move[out_gamut] = new_lab - lab_list[out_gamut] #movement in ok space
			lab_list[out_gamut] = new_lab

		return lab_list, clip_move

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
