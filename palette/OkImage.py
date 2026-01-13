import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from numba import njit

import os.path

from .OrderedDither import OrderedDither
from .OkLab import OkLab
from .OkTools import OkTools



#njit must be floating function
@njit(fastmath=True)
def _OkImage_njitFloydSteinberg(pixels:np.ndarray, pal_colors:np.ndarray, alpha_count:int, width:int, height:int, mask_weight: float, margin:float):
	i = -1
	margin_sq: float = margin*margin
	pal_len: int = pal_colors.shape[0]
	alpha_count_no_end = alpha_count - 1 #exclude end point
	alpha_count_no_end = 1 if alpha_count_no_end<1 else alpha_count_no_end #max(1,)
	alpha_step: float = 1.0/alpha_count_no_end
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
			quant_error_l = (pixels[i, 0] - pal_colors[best_idx, 0])/16.0 * mask_weight
			quant_error_a = (pixels[i, 1] - pal_colors[best_idx, 1])/16.0 * mask_weight
			quant_error_b = (pixels[i, 2] - pal_colors[best_idx, 2])/16.0 * mask_weight

			#int trunacte rounding trick
			new_alpha = 1.0
			if alpha_count>1:
				new_alpha = float( int((pixels[i, 3]+alpha_step/2.0)/alpha_step) )*alpha_step
			elif alpha_count==0:
				new_alpha = pixels[i, 3]
			alpha_error = (pixels[i, 3] - new_alpha)/16.0 * mask_weight

			#pixels = new_pixels
			pixels[i, 0] = pal_colors[best_idx, 0]
			pixels[i, 1] = pal_colors[best_idx, 1]
			pixels[i, 2] = pal_colors[best_idx, 2]
			pixels[i, 3] = new_alpha

			if y + 1 < height: #bottom
				j = i + width
				pixels[j, 0] += quant_error_l * 5.0
				pixels[j, 1] += quant_error_a * 5.0
				pixels[j, 2] += quant_error_b * 5.0
				pixels[j, 3] += alpha_error * 5.0
				if x - 1 >= 0: #bottom left
					j = i + width - 1	
					pixels[j, 0] += quant_error_l * 3.0
					pixels[j, 1] += quant_error_a * 3.0
					pixels[j, 2] += quant_error_b * 3.0
					pixels[j, 3] += alpha_error * 3.0
				if x + 1 < width: #bottom right
					j=i + width + 1
					pixels[j, 0] += quant_error_l
					pixels[j, 1] += quant_error_a
					pixels[j, 2] += quant_error_b
					pixels[j, 3] += alpha_error
			if x + 1 < width: #right
				j = i + 1
				pixels[j, 0] += quant_error_l * 7.0
				pixels[j, 1] += quant_error_a * 7.0
				pixels[j, 2] += quant_error_b * 7.0
				pixels[j, 3] += alpha_error * 7.0

	return pixels



### OkImage Sub-classes ###

@dataclass
class OkImage_Preset:
	"""Try to create valid preset, if fail OkImage.Preset.valid = False"""
	DITHER_METHOD = ("none", "bayer", "steinberg", "blue")

	image: str 				= None
	palette: str 			= None
	output: str				= None
	alpha_count: int 		= 0
	max_error: float 		= 0.0
	merge_radius: float 	= 0.0
	dither: str 			= "none"
	mask_size: int 		= 16
	mask_weight: float 	= 1.0
	print_stats: bool		= False

	valid: bool = field(default=False, init=False)

	def __post_init__(self):
		if self.image is None:
			print("Invalid OkImage.Preset.image")
			return
		if self.palette is None:
			print("Invalid OkImage.Preset.palette")
			return
		if self.output is None:
			self.output = ""

		#assign dither
		self.dither = str(self.dither).lower()
		if self.dither not in self.DITHER_METHOD:
			print("Preset fail: Invalid Preset.dither " + str(self.dither))
			return

		#assign output
		image_path = os.path.dirname(self.image)
		image_path = "./" if image_path=="" else image_path
		input_basename = os.path.basename(self.image)

		output_path = os.path.dirname(self.output)
		output_basename = os.path.basename(self.output)

		if self.output == "./":
			#current dir
			self.output = "./" + "p_" + os.path.basename(self.image)

		elif self.output=='':
			#same dir as source with p_ prefix
			self.output = os.path.join(image_path, "p_"+ input_basename)

		elif output_basename == '':
			#Only output folder provided
			self.output = os.path.join(output_path, "p_"+ input_basename)

		#preset validity check
		#print info about all invalid paths
		preset_files = [
			[self.image, os.R_OK ],
			[self.palette, os.R_OK ], 
			[self.output, os.W_OK ],
		]
		
		self.valid = OkTools.validateFileList(preset_files)



#striped list. item = (color[i],area[i])
@dataclass
class OkImage_UniqueList:
	"""Store unique colors and their mapping to OkImage.pixels"""
	color: np.ndarray = None
	area: np.ndarray = None

	unique_idxs: np.ndarray = None
	original_idxs: np.ndarray = None #colors_with_dupes = color[original_idxs]
	_tree: cKDTree = None

	def __len__(self):
		return len(self.color) if (self.color is not None) else 0

	def copy(self):
		new_copy = OkImage.UniqueList()
		new_copy.color = self.color.copy()
		new_copy.area = self.area.copy()
		new_copy.unique_idxs = self.unique_idxs.copy()
		new_copy.original_idxs = self.original_idxs.copy()
		new_copy._tree = None
		return new_copy

	def update(self, pixels):
		lab_dtype = np.dtype((np.void, pixels.dtype.itemsize * 3))
		pixels_view = pixels[:,:3].view(lab_dtype).ravel()

		unique_view, unique_idxs, original_idxs = np.unique(pixels_view, return_index=True, return_inverse=True)
		unique_colors = pixels[unique_idxs,:3]

		opaque = pixels[:, 3]
		area = np.bincount(
			original_idxs,
			weights=opaque,
			minlength=len(unique_colors)
		)

		self.color = np.ascontiguousarray(unique_colors, dtype=np.float32)
		self.area = area
		self.unique_idxs = unique_idxs
		self.original_idxs = original_idxs
		
	@property
	def tree(self):
		if self._tree is None:
			if self.color is None:
				print("OkImage.UniqueList.update(pixels) must be called before accessing its tree.")
				return None
			self._tree = cKDTree(self.color)
		return self._tree


### OkImage.Filter ###
class OkImage_Filter:
	"""Filters applied to OkImage.pixels
		Return None
	"""

	@staticmethod
	def _isValidDitherInput(image_ok, palette_ok, preset):
		if image_ok is None:
			print("Invalid image_ok")
			return False
		if palette_ok is None:
			print("Invalid palette_ok")
			return False
		if (preset is None) or (not preset.valid):
			print("Invalid preset")
			return False
		return True



	### Color manipulation ###

	@staticmethod
	def quantizeAxes(image_ok, step_count: int, axes: list[int] = [0,1,2,3] ):
		"""0=no change, 1 = set to 1.0, >1 limit steps to <step_count>"""
		if step_count == None or step_count <= 0:
			return

		if step_count <= 1:
			image_ok.pixels[:,axes] = 1.0
		else:
			step_count = step_count-1 #exclude end point
			step_size = 1.0/step_count
			#quant by precision truncation
			image_ok.pixels[:,axes] = np.floor((image_ok.pixels[:,axes] + step_size/2.0)/step_size) * step_size 


	### Filter methods ###

	@staticmethod
	def applyPalette(image_ok, unique_mapping):
		image_ok.pixels[:,:3] = unique_mapping.color[unique_mapping.original_idxs]

	@staticmethod
	def ditherNone(image_ok, palette_ok, preset):
		if not OkImage.Filter._isValidDitherInput(image_ok, palette_ok):
			return

		pal_list = palette_ok.unique_list

		_, idxs = pal_list.tree.query(image_ok.pixels[:,:3], k=1, workers=-1)
		image_ok.pixels[:,:3] = pal_list.color[:,:3][idxs]


	## Dither

	#https://bisqwit.iki.fi/story/howto/dither/jy/
	@staticmethod
	def ditherOrdered(image_ok, palette_ok, preset):
		if not OkImage.Filter._isValidDitherInput(image_ok, palette_ok, preset):
			return

		pal_list = palette_ok.unique_list
		pixels = image_ok.pixels #lab+alpha

		#Channel gaps of nearest 2 palette colors to current pixel
		pal_dists, idxs = pal_list.tree.query(pixels[:,:3], k=2, workers=-1)
		palette_gaps = np.abs(pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]]) #lab

		#preset.mask_weight rules dither
		# 0.0 to 1.0 : 0.0 -> palette_gaps
		# 1.0 to 2.0 : palette_gaps -> palette_gaps_norm
		max_pal_dist = np.max(pal_dists[:,1])
		gater = pal_dists[:,1]/(max_pal_dist+1e-12)
		gater = gater[:,None]*[1,1,1]

		if preset.mask_weight < 1.0:
			palette_gaps = OkTools.vec3Lerp(np.zeros_like(palette_gaps), palette_gaps, gater)
		else:
			palette_gaps_length = OkTools.vec3Length(palette_gaps,axis=1)[:,None]*[1,1,1]
			palette_gaps = OkTools.vec3Lerp(palette_gaps, palette_gaps_length, gater)

		#Create dither mask
		if preset.dither == "bayer":
			thresholds_stack = OrderedDither.bayerOklab(preset.mask_size)
		elif preset.dither == "blue":
			thresholds_stack = OrderedDither.blueNoiseOklab(preset.mask_size)
		else:
			thresholds_stack = OrderedDither.fallbackMatrix()
		m_l, m_a, m_b, m_o = thresholds_stack

		#Tile mask
		y_idxs, x_idxs = np.divmod(np.arange(image_ok.height*image_ok.width), image_ok.width)

		t_y_idxs = y_idxs % m_l.shape[0]
		t_x_idxs = x_idxs % m_l.shape[1]
		t_l = m_l[t_y_idxs, t_x_idxs]
		t_a = m_a[t_y_idxs, t_x_idxs]
		t_b = m_b[t_y_idxs, t_x_idxs]
		t_o = m_o[t_y_idxs, t_x_idxs]

		#thresholds lab
		thresholds_lab = np.stack((t_l, t_a, t_b), axis=1)
		pixels[:,:3]+= thresholds_lab * palette_gaps

		_, idxs = pal_list.tree.query(pixels[:,:3], k=1, workers=-1)
		pixels[:,:3] = pal_list.color[idxs]

		#thresholds alpha
		if preset.alpha_count:
			alpha_gaps = 1.0/np.float32(max(1, preset.alpha_count-1)) #-1 to exclude end point
			pixels[:, 3]+= t_o * alpha_gaps * preset.mask_weight
			OkImage.Filter.quantizeAxes(image_ok, step_count=preset.alpha_count, axes=[3])
		return pixels


	@staticmethod
	def ditherFloydSteinberg(image_ok, palette_ok, preset):
		if not OkImage.Filter._isValidDitherInput(image_ok, palette_ok, preset):
			return

		image_ok.pixels = _OkImage_njitFloydSteinberg(
			image_ok.pixels, 
			palette_ok.unique_list.color, 
			preset.alpha_count, 
			image_ok.width, 
			image_ok.height, 
			min(1.0,preset.mask_weight), #prevent runaway
			OkTools.OKLAB_8BIT_MARGIN
		)




### OkImage main class ###

class OkImage:
	#Namespacing
	Preset = OkImage_Preset
	UniqueList = OkImage_UniqueList
	Filter = OkImage_Filter

	#class vars
	pixels = None

	height = None
	width = None

	_unique_list = None

	@property
	def unique_list(self):
		if self._unique_list == None:
			self._unique_list = OkImage.UniqueList()
			self._unique_list.update(self.pixels)
		return self._unique_list

	#public methods

	## I/O
	
	def copy(self):
		new_copy = OkImage()
		new_copy.pixels = self.pixels.copy() if (self.pixels is not None) else None
		new_copy.height = self.height
		new_copy.width = self.width
		new_copy._unique_list = self._unique_list.copy() if (self._unique_list is not None) else None
		return new_copy

	def loadImage(self, input_path: str):
		in_img = Image.open(input_path).convert("RGBA")
		col_list = np.ascontiguousarray(in_img, dtype=np.float32)
		col_list = col_list.reshape(-1, 4) / 255.0
		col_list[:,:3] = OkLab.srgbToOklab(col_list[:,:3])
		ok_steps = int(np.ceil(1.0/OkTools.OKLAB_8BIT_MARGIN))

		self.pixels = col_list
		self.pixels = np.ascontiguousarray(self.pixels.copy())
		self.width, self.height = in_img.size

		OkImage.Filter.quantizeAxes(self, ok_steps, [0,1,2])

		self._unique_list = None #becomes stale

	def saveImage(self, output_path: str):
		col_list = self.pixels.copy()
		col_list[:,:3] = OkLab.oklabToSrgb(col_list[:,:3])
		rgba = np.clip(np.round(col_list * 255), 0, 255)
		rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
		rgba = rgba.reshape((self.height, self.width, 4))
		img = Image.fromarray(rgba, "RGBA")
		img.save(output_path, compress_level=1)
