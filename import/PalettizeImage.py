import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from scipy.spatial import cKDTree as KDTree
from numba import njit

import os.path
import argparse

import sys
sys.path.insert(1, './import/')
from BayerMatrix import *
from oklabConversion import *

@dataclass
class ConvertPreset:
	image: str #file names
	palette: str
	output: str
	alpha_count: int
	max_error: float #radius that within neighboring palette colors can replace unique colors
	merge_radius: float #quantize original image. >1.0 is lower quant than palette. May improve quality and performance if you got thousands of unique colors and tiny palette
	dither: int #0=None 1=ordered 2=Floyd–Steinberg(very slow)
	bayer_size: int #bayer matrix size. Only powers of two will produce proper bayer matrices

#### Image conversion ###

@njit
def OkImage_njitFloydSteinberg(pixels:np.ndarray, pal_colors:np.ndarray, width:int, height:int):
	def findClosest(col, palette_colors):
		diffs = palette_colors - col
		dists = np.sum(diffs*diffs, axis=1)
		return palette_colors[np.argmin(dists)]

	for i in range(pixels.shape[0]):
		y = i // width
		x = i - y * width

		old_pixel = pixels[i].copy()
		new_pixel = findClosest(old_pixel,pal_colors)
		pixels[i] = new_pixel
		quant_error = (old_pixel - new_pixel)/16.0

		if x + 1 < width:
			pixels[i + 1] 			+= quant_error * 7.0
		if x - 1 >= 0 and y + 1 < height:
			pixels[i + width - 1]+= quant_error	* 3.0 
		if y + 1 < height:
			pixels[i + width] 	+= quant_error * 5.0
		if x + 1 < width and y + 1 < height:
			pixels[i + width + 1]+= quant_error	* 1.0
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
 
	def __init__(self, input_path):
		self.imgToOkPixels(input_path)

	def _quantize(self, vals, step_count: int):
		step_count = int(max(1,step_count))
		return np.round(vals*step_count)/step_count

	#public
	def imgToOkPixels(self, img_path: str):
		in_img = Image.open(img_path).convert("RGBA")
		col_list = np.array(in_img, dtype=np.float64) / 255.0
		col_list = col_list.reshape(-1, 4)
		col_list[:,:3] = srgbToOklab(col_list[:,:3])
		col_list[:,:3] = self._quantize(col_list[:,:3], int(1.0/OKLAB_8BIT_MARGIN))
  
		self.pixels = col_list
		self.pixels_output = self.pixels.copy()
		self.width, self.height = in_img.size

	def saveImage(self, output_path: str):
		col_list = self.pixels_output.copy()
		col_list[:,:3] = oklabToSrgb(col_list[:,:3])
		rgba = np.clip(np.round(col_list * 255), 0, 255).astype(np.uint8)
		rgba = rgba.reshape((self.height, self.width, 4))
		img = Image.fromarray(rgba, "RGBA")
		img.save(output_path)

	def quantizeAxes(self, step_count: int):
		quant_lab = self._quantize(self.pixels_output[:,:3], step_count)
		self.pixels_output[:,:3] = quant_lab
		
	def quantizeAlpha(self, alpha_count: int):
		alpha = self.pixels_output[:,3]
		if alpha_count is None or alpha_count == 0:
			alpha = np.zeros(len(alpha)) + 1.0
		else:
			alpha = self._quantize(alpha,alpha_count)
		self.pixels_output[:,3] = alpha

	def createUniqueList(self):
		#strip dupes
		unique_colors, unique_idxs, original_idxs = np.unique(self.pixels_output, axis=0, return_index=True, return_inverse=True)
 
		#area[original_index] = dupe_count, so area[0] is how many pixels are unique_color[0]
		nontransp = self.pixels_output[:, 3] > (1.0 / 255.0) #exclude transparent
		area = np.bincount( 
	  		original_idxs, 
		 	weights=nontransp, 
		  	minlength=len(unique_colors)
		)
  
		self.unique_list = UniqueList(
			unique_colors[:,:3], 
			unique_colors[:, 3], 
	  		area,
			unique_idxs,
			original_idxs
		)
		self.unique_list.color = np.ascontiguousarray(self.unique_list.color, dtype=np.float64)

	#### palettize methods ###
	def applyPalette(self, unique_palettized):
		self.pixels_output[:,:3] = unique_palettized[self.unique_list.original_idxs]
  
	def ditherNone(self, palette_img):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3]
  
		pal_tree = pal_list.getUniqueTree()
		_, idxs = pal_tree.query(pixels, k=1)
		self.pixels_output[:,:3] = pal_list.color[:,:3][idxs]
  
  
	def ditherOrdered(self, palette_img, matrix_size=16):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3].copy()
		
		#per axis pixel gap
		pal_tree = pal_list.getUniqueTree()
		_, idxs = pal_tree.query(pixels, k=2)
		pixel_gaps = pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]] #per axis
		pixel_gaps = np.linalg.norm(pixel_gaps,axis=1)

		#Each channel uses same but differently staggered bayer matrix
		pixel_thresholds = Bayer_calcPixelThresholds(matrix_size, pixel_gaps, self.width)

		thresholds_stack = np.stack(pixel_thresholds, axis=1)
		new_pixels = pixels + thresholds_stack * pixel_gaps[:, None] 

		pal_tree = pal_list.getUniqueTree()
		_, idxs = pal_tree.query(new_pixels, k=1)
		new_pixels = pal_list.color[idxs]
  
		self.pixels_output[:,:3] = new_pixels


	def ditherFloydSteinberg(self, palette_img):
		pixels = self.pixels_output[...,:3]
		pal_colors = palette_img.unique_list.color

		pixels = OkImage_njitFloydSteinberg(pixels, pal_colors, self.width, self.height)

		self.pixels_output[...,:3] = pixels



#Map unique colors to palette, but avoid collapsing similar colors
#Return unique_palettized[len(unique_list.color)] = [l,a,b,alpha]
def Palettize_createWeighted(
	src_img: OkImage, 
	palette_img: OkImage,
	max_error: int = 1,
	k_count = 13
):
	def calcBucketScore(bucket_areas, col_dists, col_idxs, max_radius):
		area_weight = max(8.0,4.0*max_radius)
		max_bucket = max(1.0, np.max(bucket_areas))
		return col_dists * (1.0 + area_weight * (bucket_areas[col_idxs]/max_bucket) )

	unique_list = src_img.unique_list
	palette_list = palette_img.unique_list
	pal_length = len(palette_list.color)

	max_radius = approxOkGap(pal_length) * max_error
 
	#accumulated area of colors in each palette bucket
	bucket_areas = np.zeros(pal_length)

	#Closest palette colors
	pal_tree = palette_list.getUniqueTree()
	est_maxk = max_error * 12 #Sphere kissing number within 1 radius
	k_count = max(2, min(pal_length,est_maxk) )
	dists, idxs = pal_tree.query(unique_list.color, k = int(k_count))
  	
	#choose palette index for each color
	unique_count = len(unique_list.color)
	unique_palettized = np.zeros((unique_count,3))

	#prioritize largest area
	unique_sorted_idx = np.argsort(-1.0*unique_list.area)
	for i in unique_sorted_idx:
		#lowest dist and emptiest bucket ; lowest score = better
		local_scores = calcBucketScore(bucket_areas, dists[i], idxs[i], max_radius)
		mask = dists[i] <= max_radius
	
		if np.any(mask):
			valid = np.where(mask)[0]
			best_pos = valid[np.argmin(local_scores[valid])]
			best_j = int(idxs[i][best_pos])
		else:
			#choose nearest if exceeds max_error
			best_pos = 0
			best_j = int(idxs[i][best_pos])
	
		unique_palettized[i] = palette_list.color[best_j]
		bucket_areas[best_j] += unique_list.area[i]
  
	return unique_palettized
	

### Palettize Image ###

def Palettize_preset(preset: ConvertPreset):
	
	palette_ok = OkImage(preset.palette)
	palette_ok.quantizeAlpha(None)
	palette_ok.createUniqueList()
	pal_length = len(palette_ok.unique_list.color)
	if pal_length < 2:
		print("At least 2 palette colors needed.")
		return
 
	image_ok = OkImage(preset.image)
	if preset.merge_radius:
		axis_step_size = approxOkGap(pal_length) * preset.merge_radius
		axis_count = int(1.0/axis_step_size)
		image_ok.quantizeAxes(axis_count)
	image_ok.quantizeAlpha(preset.alpha_count)
	image_ok.createUniqueList()

	#replace original img pixels with convert_dict
	if preset.dither == 1:
		image_ok.ditherOrdered(palette_ok,preset.bayer_size)
	elif preset.dither == 2:
		image_ok.ditherFloydSteinberg(palette_ok)
	else: #0
		if preset.max_error:
			unique_palettized = Palettize_createWeighted(image_ok, palette_ok, preset.max_error)
			image_ok.applyPalette(unique_palettized)
		else:
			image_ok.ditherNone(palette_ok)

	output_path = os.path.dirname(preset.output)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	image_ok.saveImage(preset.output)
	print("Saved image "+preset.output) 
 

def Palettize_demo():
	print("Converting demo images")
   
	input_path = "./demoImages"
	palette_path = "./palettes"
	output_path = "./output"
	demo_preset_list = [
		ConvertPreset(
			image				= input_path+"/"+"KaelygonLogo25.png",
			palette			= palette_path+"/"+"pal16.png",
			output			= output_path+"/"+"p_KaelygonLogo25.png",
			alpha_count		= 1,
			max_error		= 2.0,
			merge_radius	= 0.0,
			dither			= 0,
			bayer_size		= 0,
		),
		ConvertPreset(
			image				= input_path+"/"+"LPlumocrista.png",
			palette			= palette_path+"/"+"pal256.png",
			output			= output_path+"/"+"p_LPlumocrista.png",
			alpha_count		= 1,
			max_error		= 0.0,
			merge_radius	= 0.0,
			dither			= 1,
			bayer_size		= 16,
		),
		ConvertPreset(
			image				= input_path+"/"+"Michelangelo_David.png",
			palette			= palette_path+"/"+"wplacePalette.png",
			output			= output_path+"/"+"p_Michelangelo_David.png",
			alpha_count		= 1,
			max_error		= 1.0,
			merge_radius	= 0.1,
			dither			= 0,
			bayer_size		= 0,
		),
		ConvertPreset(
			image				= input_path+"/"+"gray.png",
			palette			= palette_path+"/"+"pal2.png",
			output			= output_path+"/"+"p_gray.png",
			alpha_count		= 256,
			max_error		= 1.0,
			merge_radius	= 0.0,
			dither			= 1,
			bayer_size		= 16,
		)
	]
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)
 
	for preset in demo_preset_list:
		Palettize_preset( preset )

	print("Demo done!")

def Palettize_parser(argv):

	parser = argparse.ArgumentParser(prog=argv[0],description="Palettize and dither using arbitrary palette")
 
	#flagged
	parser.add_argument(
		'-i', '--input', type=str,
  		default= None,
		help="input .png path"	
	)
	parser.add_argument(
		'-p', '--palette', type=str,
  		default= None,
		help="Palette .png path"	 
	) 
	parser.add_argument(
		'-o', '--output', type=str,
		default=None,
		help="Output .png path"	
	) 
	parser.add_argument(
		'-a', '--alpha-count', type=int,
		default=1,
  		help="Number of alpha levels"
	) 
	parser.add_argument(
		'-e', '--max-error', type=float,
		default=0.0,
		help="Radius that within neighboring palette colors can replace unique colors. Works only with --dither none"
	) 
	parser.add_argument(
		'-m', '--merge-radius', type=float,
		default=0.0,
		help="Quantize before palettization. 1.0 is roughly same number of colors as palette."
	) 
	parser.add_argument(
		'-d', '--dither', type=str,
  		default="none",
  		help="Options: none, bayer, steinberg"
	) 
	parser.add_argument(
		'-b', '--bayer-size', type=int,
  		default=16,
  		help="Dither matrix size. Powers of 2 are bayer matrices. Works only with --dither bayer"
	) 
	parser.add_argument(
	 	'-D', '--demo',  type=str,
		default="False", 
		dest='demo', 
		help="Generate test images"
	)

	if len(argv)<=1:
		argv.append("--help")
	args = parser.parse_args(argv[1:])

	#demo has highest priority
	if args.demo is not None:
		if args.demo.upper() in ["TRUE", "1"]:
			Palettize_demo()
			return None

	#failures
	if args.input is None:
		print("Must specify input image!")
		return None
	if args.palette is None:
		print("Must specify palette image!")
		return None

	#Args to preset
	d_preset = ConvertPreset(
		image				= str(args.input),
		palette			= str(args.palette),
		output			= args.output,
		alpha_count		= int(args.alpha_count),
		max_error		= float(args.max_error),
		merge_radius	= float(args.merge_radius),
		dither			= None,
		bayer_size		= int(args.bayer_size),
	)
 
	dither_options = {
		"NONE" : 0,
		"BAYER" : 1,
		"ORDERED" : 1,
		"STEINBERG" : 2,
		"FLOYD-STEINBERG" : 2,
		"0" : 0,
		"1" : 1,
		"2" : 2,
	}
 
	args_dither_upper = str(args.dither).upper()

	if args.dither is not None and args_dither_upper in dither_options:
		d_preset.dither = dither_options[args_dither_upper]
	else:
		d_preset.dither = 0

	if args.output==None:
		f_name = os.path.basename(d_preset.image)
		f_path = os.path.dirname(d_preset.image)
		d_preset.output=f_path+"/"+"p_"+ f_name
	elif d_preset.output=="./":
		d_preset.output="./"+"p_"+os.path.basename(d_preset.image)
  
	return d_preset