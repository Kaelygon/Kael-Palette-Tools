"""Namespace for misc tools used by palette"""

import os
import numpy as np
from .OkLab import OkLab

#Manipulate arrays of colors
class OkTools:
	### Constants ###
	FALLBACK_NORM = np.array([ 0.57735027 ]*3) * [1,-1,1] #sqrt(1/3)

	OKLAB_8BIT_MARGIN =  7.011e-05  # minimum SRGB distance in oklab space
	OKLAB_GAMUT_VOLUME =  0.05356533  # (oklab gamut) / (srgb gamut)

	OKLAB_BOX_MIN =   np.array( [ 0.        , -0.23388757, -0.31152815] ) # OkLab bounding box
	OKLAB_BOX_MAX =   np.array( [0.99999999, 0.27456629, 0.19856975] )
	OKLAB_BOX_SIZE = np.array( [0.99999999, 0.50845387, 0.5100979 ] )

	DARKEST_BLACK_LAB = OkLab.srgbToOklab(np.array([[0.499/255,0.499/255,0.499/255]]))[0] #brighest 8-bit SRGB that rounds to pure black 


	### Vec3 Tools ###

	@staticmethod
	def vec3Length(vector_list, axis=-1, keepdims=False):
		"""float[...,1] vec3Length(float[...,3] vector_list, int axis=1, bool keepdims=false)"""
		sq_sum = np.einsum('...i,...i->...', vector_list, vector_list)
		if keepdims:
			sq_sum = np.expand_dims(sq_sum, axis=axis)
		return np.sqrt(sq_sum)

	@staticmethod
	def vec3ArrayNorm(vector_list, axis=-1):
		"""float[...,3] vec3ArrayNorm(float[...,3] vector_list, int axis=1, bool keepdims=false)"""
		l = OkTools.vec3Length(vector_list, axis=axis, keepdims=True)
		norm = vector_list / (l + 1e-12)
		norm[np.all(vector_list == 0, axis=axis)] = OkTools.FALLBACK_NORM
		return norm, np.squeeze(l, axis=-1)

	# vec_b -> vec_a : fac(0 -> 1)
	@staticmethod
	def vec3Lerp(vec_a, vec_b, fac):
		return vec_a * (1.0-fac) + vec_b * fac

	#Generate surface normals of n points evenly distributed on a sphere
	#offset (n,m) shifts the points on surface
	@staticmethod
	def sphereNormals(point_count: int, offset: np.ndarray = 0.0):
		offset = np.atleast_1d(offset).astype(float)
		golden_ratio = (1 + 5**0.5) / 2.0
		powers = golden_ratio ** np.arange(offset.shape[-1]) #irrational base polynomial
		angle_offset = np.dot(offset, powers) #x * b**2 + y * b**1 + z * b**2

		indices = np.arange(0, point_count, dtype=float) + 0.5
		alpha = np.arccos(1 - 2*indices/point_count)
		theta = 2 * np.pi * golden_ratio * (indices + angle_offset)
		
		x = np.sin(alpha) * np.cos(theta)
		y = np.sin(alpha) * np.sin(theta)
		z = np.cos(alpha)

		return np.stack((x, y, z), axis=1)



	### Color Tools ###

	@staticmethod
	def inOklabGamut(lab_list, eps = 1e-12, lower_bound = 0.0, upper_bound = 1.0, axis=-1):
		"""bool[] inOklabGamut(float[][3] lab_list, float eps = 1e-12, float lower_bound = 0.0, float upper_bound = 1.0 ))"""
		lin_list = OkLab.oklabToLinear(lab_list)
		in_gamut = (lin_list >= lower_bound-eps) & (lin_list <= upper_bound+eps)
		in_gamut = in_gamut.all(axis=axis)
		return in_gamut

	@staticmethod
	def clipToOklabGamut(lab_list, eps = 1e-12, lower_bound = 0.0, upper_bound = 1.0, axis=-1):
		"""(float[][3] float[][3]) clipToOklabGamut(float[][3] lab_list, float eps = 1e-12, float lower_bound = 0.0, float upper_bound))"""
		lin_list = OkLab.oklabToLinear(lab_list)
		out_gamut = (lin_list <= lower_bound-eps) | (lin_list >= upper_bound+eps)
		out_gamut = out_gamut.any(axis=axis)

		if not np.any(out_gamut):
			return lab_list, None

		new_pos = np.clip(lin_list[out_gamut],[0.0]*3,[1.0]*3)
		new_lab = OkLab.linearToOklab(new_pos)

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
		rgb_list = OkLab.oklabToSrgb(lab_list)
		is_gray = (
			(abs(rgb_list[:,0]-rgb_list[:,1]) < threshold) & 
			(abs(rgb_list[:,1]-rgb_list[:,2]) < threshold)
		)
		return is_gray



	### Misc tools ###

	@staticmethod
	def srgbToHex(rgb):
		"""char* srgbToHex(float[3] rgb)"""
		rgb = np.clip(rgb,[0.0]*3,[1.0]*3)
		rgb = np.round(rgb * 255.0)	
		rgb = rgb.astype(np.uint8)
		return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])

	@staticmethod
	def approxOkGap(point_count: int):
		return (OkTools.OKLAB_GAMUT_VOLUME/max(1,point_count))**(1.0/3.0)

	@staticmethod
	def validateFileList(file_list: list[[str,str]]):
		preset_success = True
		for file, access_flag in file_list:

			if (file is None) or (file==''):
				print("Undefined file")
				preset_success = False
				continue

			#directory
			base_dir = os.path.dirname(file)
			base_dir = "./" if base_dir=='' else base_dir
			if not os.path.isdir(base_dir):
				print("Directory doesn't exist " + base_dir)
				preset_success = False
				continue
			if not os.access(base_dir, access_flag):
				print("Can't access directory "+base_dir)
				preset_success = False
				continue

			#file
			if (access_flag == os.R_OK):
				if not os.path.exists(file):
					print("File doesn't exist "+file)
					preset_success = False
			else:
				if os.path.exists(file):
					#writable file exists
					if not os.access(file, access_flag):
						print("Can't access file "+file)
						preset_success = False
				else:
					#writable doesn't exists
					if not os.access(base_dir, access_flag):
						print("Can't create file "+file)
						preset_success = False
		return preset_success