import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from scipy.spatial import cKDTree as KDTree
from numba import njit

import os.path
import argparse

import sys
sys.path.insert(1, './import/')
from OrderedDither import *
from oklabConversion import *

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

#### Image conversion ###

@njit
def OkImage_njitFloydSteinberg(pixels:np.ndarray, pal_colors:np.ndarray, width:int, height:int):
	for i in range(pixels.shape[0]):
		y = i // width
		x = i - y * width

		old_pixel = pixels[i].copy()

		diffs = pal_colors - old_pixel
		dists = np.sum(diffs*diffs, axis=1)
		new_pixel = pal_colors[np.argmin(dists)]

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
		_, idxs = pal_tree.query(new_pixels, k=1)
		new_pixels = pal_list.color[idxs]
		return new_pixels


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
		if alpha_count == None:
			return

		alpha = self.pixels_output[:,3]
		if alpha_count == 0:
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

	#https://bisqwit.iki.fi/story/howto/dither/jy/
	def ditherOrdered(self, palette_img, matrix_size=16, dither_weight=1.0):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3].copy()
		pal_tree = pal_list.getUniqueTree()

		#Channel gaps of nearest 2 palette colors to current pixel
		pal_dists, idxs = pal_tree.query(pixels, k=2)
		palette_gaps = np.abs(pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]])

		#scale by gap norm weighted by distance
		if dither_weight!=0.0:
			# vec_b -> vec_a : fac(0 -> 1)
			def np_lerp(vec_a, vec_b, fac):
				return vec_a * (1.0-fac) + vec_b * fac

			palette_gaps_norm = np.linalg.norm(palette_gaps,axis=1)[:,None]*[1,1,1] #smoothest and best colors but over-dithers with some palettes

			#limit over-dither
			max_pal_dist = np.max(pal_dists[:,0])
			gater = pal_dists[:,0]/max_pal_dist
			gater = gater + max(0.0, dither_weight - 1.0) #1.0-2.0 raises minimum
			gater = np.clip(gater * dither_weight, 0.0, 1.0)

			palette_gaps = np_lerp(palette_gaps, palette_gaps_norm, gater[:,None]*[1,1,1])

		thresholds_lab = ordered.bayerOklab(matrix_size)

		self.pixels_output[:,:3] = self._applyDitherThresholds(pixels, thresholds_lab, palette_gaps, pal_list)

	def ditherBlue(self, palette_img, matrix_size=16, dither_weight=1.0):
		pal_list = palette_img.unique_list
		pixels = self.pixels_output[:,:3].copy()
		pal_tree = pal_list.getUniqueTree()

		pal_dists, idxs = pal_tree.query(pixels, k=2)
		palette_gaps = np.abs(pal_list.color[idxs[:,1]] - pal_list.color[idxs[:,0]])

		blue_thresholds = ordered.blueNoiseOklab(matrix_size,matrix_size)
		blue_thresholds = np.array(blue_thresholds) * dither_weight

		self.pixels_output[:,:3] = self._applyDitherThresholds(pixels, blue_thresholds, palette_gaps, pal_list)


	def ditherFloydSteinberg(self, palette_img):
		pixels = self.pixels_output[...,:3]
		pal_colors = palette_img.unique_list.color

		pixels = OkImage_njitFloydSteinberg(pixels, pal_colors, self.width, self.height)

		self.pixels_output[...,:3] = pixels

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
		total_bias = np.linalg.norm( np.mean( quant_delta[:,:3] ) ) #vector length of mean delta
		print("[L,a,b] bias: " +
			str(round(lum_bias,4)) + ", " +
			str(round(a_bias,4)) + ", " +
			str(round(b_bias,4))
		)
		print("Alpha bias: " + str(round(alpha_bias,4)) + ", " )
		print("Total bias: " + str(round(total_bias,8)))


#Map unique colors to palette, but avoid collapsing similar colors
#Return unique_palettized[len(unique_list.color)] = [l,a,b,alpha]
def Palettize_createWeighted(
	src_img: OkImage,
	palette_img: OkImage,
	max_error: int = 1,
	k_count = 13
):
	def calcBucketScore(bucket_areas, col_dists, col_idxs, max_radius):
		#lower=better, col dists<1.0, bucket_fullness [0,1]
		area_weight = 10.0 #Arbitrary, bias filled buckets to score worse
		area_weight = max(area_weight,5.0*max_radius) #Allows high max-error to still have an effect
		max_bucket = max(1.0, np.max(bucket_areas))
		area_score = np.maximum(area_weight * bucket_areas[col_idxs]/max_bucket, 1.0)
		return col_dists * area_score

	unique_list = src_img.unique_list
	palette_list = palette_img.unique_list
	pal_length = len(palette_list.color)

	max_radius = approxOkGap(pal_length) * max_error

	#accumulated area of colors in each palette bucket
	bucket_areas = np.zeros(pal_length)

	#Closest palette colors
	pal_tree = palette_list.getUniqueTree()
	est_maxk = 0.74*(1.0+max_error)**3 + 1.0 #How many palette colors within max_error
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
	palette_ok.quantizeAlpha(0)
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
	if preset.dither == "bayer":
		image_ok.ditherOrdered(palette_ok, preset.mask_size, preset.mask_weight)

	elif preset.dither == "blue":
		image_ok.ditherBlue(palette_ok, preset.mask_size, preset.mask_weight)

	elif preset.dither == "steinberg":
		image_ok.ditherFloydSteinberg(palette_ok)

	elif preset.dither == "none":
		if preset.max_error:
			#choose closest within max_error weighted by area
			unique_palettized = Palettize_createWeighted(image_ok, palette_ok, preset.max_error)
			image_ok.applyPalette(unique_palettized)
		else:
			#choose closest to palette
			image_ok.ditherNone(palette_ok)
	
	else:
		print("Somehow dither method went missing.")
		exit(-1)

	output_path = os.path.dirname(preset.output)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	image_ok.saveImage(preset.output)
	print("Saved image "+preset.output)

	print("\nColor quant-error")
	image_ok.printImgError()


## Palettize Image Parser ##

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
			dither			= "none",
			mask_size		= 0,
			mask_weight		= 1.0,
		),
		ConvertPreset(
			image				= input_path+"/"+"LPlumocrista.png",
			palette			= palette_path+"/"+"pal256.png",
			output			= output_path+"/"+"p_LPlumocrista.png",
			alpha_count		= None,
			max_error		= 0.0,
			merge_radius	= 0.0,
			dither			= "bayer",
			mask_size		= 16,
			mask_weight		= 1.0,
		),
		ConvertPreset(
			image				= input_path+"/"+"Michelangelo_David.png",
			palette			= palette_path+"/"+"wplacePalette.png",
			output			= output_path+"/"+"p_Michelangelo_David.png",
			alpha_count		= 1,
			max_error		= 1.0,
			merge_radius	= 0.1,
			dither			= "steinberg",
			mask_size		= 0,
			mask_weight		= 1.0,
		),
		ConvertPreset(
			image				= input_path+"/"+"gray.png",
			palette			= palette_path+"/"+"pal2.png",
			output			= output_path+"/"+"p_gray.png",
			alpha_count		= 1,
			max_error		= 1.0,
			merge_radius	= 0.0,
			dither			= "bayer",
			mask_size		= 16,
			mask_weight		= 1.0,
		),
		ConvertPreset(
			image				= input_path+"/"+"rgba24_color_test.png",
			palette			= palette_path+"/"+"pal256.png",
			output			= output_path+"/"+"p_rgba24_color_test.png",
			alpha_count		= 16,
			max_error		= 0.0,
			merge_radius	= 0.0,
			dither			= "blue",
			mask_size		= 128,
			mask_weight		= 0.5,
		)
	]
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for preset in demo_preset_list:
		Palettize_preset( preset )

	print("Demo done!")

def Palettize_parser(argv):

	parser = argparse.ArgumentParser(prog=argv[0],description="Palettize and dither using arbitrary palette")

	#all inputs arg_list are strings. Easier to convert str to X
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
		'-a', '--alpha-count', type=str,
		default="1",
		help="Number of alpha levels"
	)
	parser.add_argument(
		'-e', '--max-error', type=str,
		default="0.0",
		help="Radius that within neighboring palette colors can replace unique colors. Works only with --dither none"
	)
	parser.add_argument(
		'-mr', '--merge-radius', type=str,
		default="0.0",
		help="Quantize before palettization. 1.0 is roughly same number of colors as palette."
	)
	parser.add_argument(
		'-d', '--dither', type=str,
		default="none",
		help="Options: none, bayer, steinberg, blue"
	)
	parser.add_argument(
		'-ms', '--mask-size', type=str,
		default=16,
		help="Dither matrix size for dither=(bayer, blue)"
	)
	parser.add_argument(
		'-dw', '--mask-weight', type=str,
		default=16,
		help="Dither strength for dither=(none, bayer)"
	)
	parser.add_argument(
		'-D', '--demo',  type=str,
		default="False",
		dest='demo',
		help="Generate test images"
	)

	if len(argv)<=1: #No args
		argv.append("--help")
	arg_list = parser.parse_args(argv[1:])

	#demo has highest priority
	if (arg_list.demo is not None) and (str(arg_list.demo).lower() in ["true", "1"]):
		Palettize_demo()
		return None


	#arg_list to preset
	d_preset = ConvertPreset(
		image				= str(arg_list.input),
		palette			= str(arg_list.palette),
		output			= None,
		alpha_count		= int(arg_list.alpha_count),
		max_error		= float(arg_list.max_error),
		merge_radius	= float(arg_list.merge_radius),
		dither			= None,
		mask_size		= int(arg_list.mask_size),
		mask_weight		= float(arg_list.mask_weight),
	)


	#assign dither
	arg_list.dither = str(arg_list.dither).lower()
	if arg_list.dither in ConvertPreset.DITHER_METHOD:
		method_index = ConvertPreset.DITHER_METHOD[arg_list.dither]
		d_preset.dither = ConvertPreset.DITHER_METHOD_KEYS[ method_index ]
	else:
		print("Invalid dither method " + str(d_preset.dither))
		print("Defaulting to \"none\"")
		d_preset.dither = "none"


	#assign output
	if arg_list.input != None:
		intput_path = os.path.dirname(arg_list.input)
		intput_basename = os.path.basename(arg_list.input)
	if arg_list.output != None:
		output_path = os.path.dirname(arg_list.output)
		output_basename = os.path.basename(arg_list.output)

	if arg_list.output==None or arg_list.output=='':
		#same dir as source with p_ prefix
		d_preset.output = intput_path + "/" + "p_"+ intput_basename

	elif output_basename == '':
		#Only output folder provided
		d_preset.output = output_path + "/" + "p_"+ intput_basename

	elif arg_list.output == "./":
		#current dir
		d_preset.output = "./" + "p_" + os.path.basename(d_preset.image)

	else:
		d_preset.output = arg_list.output


	#preset validity check
	preset_fail = 0
	input_files = [d_preset.image, d_preset.palette ]
	preset_files = [d_preset.image, d_preset.palette, d_preset.output ]
	for file in preset_files:
		base_dir = os.path.dirname(file)
		if not os.path.isdir(base_dir):
			print("Directory "+base_dir+" doesn't exist.")
			preset_fail = 1
		if not os.access(base_dir, os.W_OK):
			print("Can't access "+base_dir)
			preset_fail = 1

	for file in input_files:
		if not os.path.exists(file):
			print("File doesn't exist "+file)
			preset_fail = 1

	if preset_fail:
		print("Failed to parse arguments. Exiting...")
		return None

	return d_preset
