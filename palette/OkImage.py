import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from scipy.spatial import cKDTree as KDTree
from numba import njit

import os.path

from palette.OrderedDither import *
from palette.OkLab import *
from palette.OkTools import *

@dataclass
class ConvertPreset:
	DITHER_METHOD = {
		"none" : 0,
		"bayer" : 1,
		"steinberg" : 2,
		"blue" : 3,
		"0" : 0,
		"1" : 1,
		"2" : 2,
		"3" : 3,
	}
	DITHER_METHOD_KEYS = [
		"none",
		"bayer",
		"steinberg",
		"blue",
	]

	image: str 				= None
	palette: str 			= None
	output: str				= None
	alpha_count: int 		= 1
	max_error: float 		= 0.0
	merge_radius: float 	= 0.0
	dither: int 			= 0
	mask_size: int 		= 16
	mask_weight: float 	= 1.0
	print_stats: bool		= False

#### Image conversion ###

@njit(fastmath=True)
def OkImage_njitFloydSteinberg(pixels:np.ndarray, pal_colors:np.ndarray, width:int, height:int, margin:float):
	i=-1
	margin_sq = margin*margin
	pal_len = pal_colors.shape[0]
	for y in range(height):
		for x in range(width):
			i+=1

			min_dist_sq = 1e10
			best_idx = 0
			for j in range(pal_len):
				diff_l = pal_colors[j, 0] - pixels[i, 0]
				diff_a = pal_colors[j, 1] - pixels[i, 1]
				diff_b = pal_colors[j, 2] - pixels[i, 2]
				dist_sq = diff_l*diff_l + diff_a*diff_a + diff_b*diff_b
				if dist_sq < min_dist_sq:
					min_dist_sq = dist_sq
					best_idx = j
					if dist_sq <= margin_sq:
						break

			#quant_error = (old_pixels - new_pixels)/16.0
			quant_error_l = (pixels[i, 0] - pal_colors[best_idx, 0])/16.0
			quant_error_a = (pixels[i, 1] - pal_colors[best_idx, 1])/16.0
			quant_error_b = (pixels[i, 2] - pal_colors[best_idx, 2])/16.0

			#pixels = new_pixels
			pixels[i, 0] = pal_colors[best_idx, 0]
			pixels[i, 1] = pal_colors[best_idx, 1]
			pixels[i, 2] = pal_colors[best_idx, 2]

			if y + 1 < height: #bottom
				j = i + width
				pixels[j, 0] += quant_error_l * 5.0
				pixels[j, 1] += quant_error_a * 5.0
				pixels[j, 2] += quant_error_b * 5.0
				if x - 1 >= 0: #bottom left
					j = i + width - 1	
					pixels[j, 0] += quant_error_l * 3.0
					pixels[j, 1] += quant_error_a * 3.0
					pixels[j, 2] += quant_error_b * 3.0
				if x + 1 < width: #bottom right
					j=i + width + 1
					pixels[j, 0] += quant_error_l
					pixels[j, 1] += quant_error_a
					pixels[j, 2] += quant_error_b
			if x + 1 < width: #right
				j = i + 1
				pixels[j, 0] += quant_error_l * 7.0
				pixels[j, 1] += quant_error_a * 7.0
				pixels[j, 2] += quant_error_b * 7.0

	return pixels


#striped list. item = (color[i],alpha[i],area[i])
@dataclass
class UniqueList:
	color: np.ndarray #uniques only
	alpha: np.ndarray
	area: np.ndarray

	unique_idxs: np.ndarray
	original_idxs: np.ndarray #colors_with_dupes = color[original_idxs]
	tree: KDTree = None

	def getUniqueTree(self):
		if self.tree is None:
			self.tree = KDTree(self.color)
		return self.tree

class OkImage:
	pixels = None #don't mutate after init
	pixels_output = None #copy of pixels that can be modified

	height = None
	width = None

	unique_list = None

	#private
	def __init__(self, input_path):
		self.imgToOkPixels(input_path)

	def _quantize(self, vals, step_count: int):
		step_count = int(max(1,step_count))
		return np.round(vals*step_count)/step_count

	def _applyDitherThresholds(self, pixels, thresholds_lab, palette_gaps, pal_list: UniqueList):
		m_l, m_a, m_b = thresholds_lab
		y_idxs, x_idxs = np.divmod(np.arange(self.height*self.width), self.width)

		t_y_idxs = y_idxs % m_l.shape[0]
		t_x_idxs = x_idxs % m_l.shape[1]
		t_l = m_l[t_y_idxs, t_x_idxs]
		t_a = m_a[t_y_idxs, t_x_idxs]
		t_b = m_b[t_y_idxs, t_x_idxs]

		thresholds_stack = np.stack((t_l, t_a, t_b), axis=1)
		new_pixels = pixels + thresholds_stack * palette_gaps

		pal_tree = pal_list.getUniqueTree()
		_, idxs = pal_tree.query(new_pixels, k=1, workers=-1)
		new_pixels = pal_list.color[idxs]
		return new_pixels


	#public
	def imgToOkPixels(self, img_path: str):
		in_img = Image.open(img_path).convert("RGBA")
		col_list = np.ascontiguousarray(in_img, dtype=np.float32)
		col_list = col_list.reshape(-1, 4) / 255.0
		col_list[:,:3] = OkLab.srgbToOklab(col_list[:,:3])
		col_list[:,:3] = self._quantize(col_list[:,:3], int(1.0/OkTools.OKLAB_8BIT_MARGIN))

		self.pixels = col_list
		self.pixels_output = self.pixels.copy()
		self.width, self.height = in_img.size

	def saveImage(self, output_path: str):
		col_list = self.pixels_output.copy()
		col_list[:,:3] = OkLab.oklabToSrgb(col_list[:,:3])
		rgba = np.clip(np.round(col_list * 255), 0, 255)
		rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
		rgba = rgba.reshape((self.height, self.width, 4))
		img = Image.fromarray(rgba, "RGBA")
		img.save(output_path, compress_level=1)

	def quantizeAxes(self, step_count: int):
		quant_lab = self._quantize(self.pixels_output[:,:3], step_count)
		self.pixels_output[:,:3] = quant_lab
		
	def quantizeAlpha(self, alpha_count: int):
		if alpha_count == None:
			return

		alpha = self.pixels_output[:,3]
		if alpha_count == 0:
			alpha = np.zeros(len(alpha)) + 1.0
		else:
			alpha = self._quantize(alpha,alpha_count)
		self.pixels_output[:,3] = alpha

	def createUniqueList(self):
		pixels = self.pixels_output
		lab_dtype = np.dtype((np.void, pixels.dtype.itemsize * 3))
		pixels_view = pixels[:,:3].view(lab_dtype).ravel()

		unique_view, unique_idxs, original_idxs = np.unique(pixels_view, return_index=True, return_inverse=True)
		unique_colors = pixels[unique_idxs,:3]

		opaque = pixels[:, 3] > (0.51 / 255.0)
		area = np.bincount(
			original_idxs,
			weights=opaque,
			minlength=len(unique_colors)
		)

		self.unique_list = UniqueList(
			unique_colors,
			pixels[unique_idxs,3],
			area,
			unique_idxs,
			original_idxs
		)
		self.unique_list.color = np.ascontiguousarray(self.unique_list.color, dtype=np.float32)

	#### palettize methods ###
	def applyPalette(self, unique_palettized):
		self.pixels_output[:,:3] = unique_palettized[self.unique_list.original_idxs]

	def ditherNone(self, palette_img):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3]

		pal_tree = pal_list.getUniqueTree()
		_, idxs = pal_tree.query(pixels, k=1, workers=-1)
		self.pixels_output[:,:3] = pal_list.color[:,:3][idxs]

	#https://bisqwit.iki.fi/story/howto/dither/jy/
	def ditherOrdered(self, palette_img, matrix_size=16, dither_weight=1.0):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3]
		pal_tree = pal_list.getUniqueTree()

		#Channel gaps of nearest 2 palette colors to current pixel
		pal_dists, idxs = pal_tree.query(pixels, k=2, workers=-1)
		palette_gaps = np.abs(pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]])

		#scale by gap norm weighted by distance
		if dither_weight!=0.0:
			# vec_b -> vec_a : fac(0 -> 1)
			def np_lerp(vec_a, vec_b, fac):
				return vec_a * (1.0-fac) + vec_b * fac

			palette_gaps_norm = OkTools.vec3Length(palette_gaps,axis=1)[:,None]*[1,1,1] #smoothest and best colors but over-dithers with some palettes

			#limit over-dither
			max_pal_dist = np.max(pal_dists[:,0])
			gater = pal_dists[:,0]/max_pal_dist
			gater = gater + max(0.0, dither_weight - 1.0) #1.0-2.0 raises minimum
			gater = np.clip(gater * dither_weight, 0.0, 1.0)

			palette_gaps = np_lerp(palette_gaps, palette_gaps_norm, gater[:,None]*[1,1,1])

		thresholds_lab = OrderedDither.bayerOklab(matrix_size)

		self.pixels_output[:,:3] = self._applyDitherThresholds(pixels, thresholds_lab, palette_gaps, pal_list)

	def ditherBlue(self, palette_img, matrix_size=16, dither_weight=1.0):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3]
		pal_tree = pal_list.getUniqueTree()

		pal_dists, idxs = pal_tree.query(pixels, k=2, workers=-1)
		palette_gaps = np.abs(pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]])

		blue_thresholds = OrderedDither.blueNoiseOklab(matrix_size,matrix_size)
		blue_thresholds = np.array(blue_thresholds) * dither_weight

		self.pixels_output[:,:3] = self._applyDitherThresholds(pixels, blue_thresholds, palette_gaps, pal_list)


	def ditherFloydSteinberg(self, palette_img):
		pixels = self.pixels_output[...,:3]
		pal_colors = palette_img.unique_list.color

		self.pixels_output[...,:3] = OkImage_njitFloydSteinberg(pixels, pal_colors, self.width, self.height, OkTools.OKLAB_8BIT_MARGIN)

	def printImgError(self):
		quant_delta = self.pixels_output - self.pixels
		dl = quant_delta[:,0]
		da = quant_delta[:,1]
		db = quant_delta[:,2]
		dalpha = quant_delta[:,3]

		#_rmsq = root mean square
		lum_rmsq 	= np.sqrt( np.mean(dl**2) )
		chroma_rmsq= np.sqrt( np.mean(da**2+db**2) )
		#alpha_rmsq = np.sqrt( np.mean(dalpha**2) )
		total_rmsq = np.sqrt( np.mean(dl**2 + da**2 + db**2) )

		print("[L,chroma] rmsq: " +
			str(round(lum_rmsq,4)) + ", " +
			str(round(chroma_rmsq,4))
		)
		#print("Alpha rmsq: " + str(round(alpha_rmsq,4)) + ", " )
		print("Total rmsq: " 	+ str(round(total_rmsq	,8)))

		lum_bias = np.mean(dl)
		a_bias = np.mean(da)
		b_bias = np.mean(db)
		alpha_bias = np.mean(dalpha)
		quant_mean = np.mean( quant_delta[:,:3], axis=0 )
		total_bias = OkTools.vec3Length( quant_mean ) #vector length of mean delta
		print("[L,a,b] bias: " +
			str(round(lum_bias,4)) + ", " +
			str(round(a_bias,4)) + ", " +
			str(round(b_bias,4))
		)
		print("Alpha bias: " + str(round(alpha_bias,4)) + ", " )
		print("Total bias: " + str(round(total_bias,8)))
