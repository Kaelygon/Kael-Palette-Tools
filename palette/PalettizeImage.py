"""Run OkImage.Preset through OkImage  """
import numpy as np
from scipy.spatial import cKDTree

import os.path
import argparse

from .OkImage import OkImage
from .OkTools import OkTools



class PalettizeImage:

	# Private
	@staticmethod
	def _strToBool(s):
		return True if str(s).lower() in ["true", "1"] else False

	@staticmethod
	def _demo():
		print("Converting demo images")
		
		input_path = "./demoImages"
		palette_path = "./palettes"
		output_path = "./output"
		demo_preset_list = [
			OkImage.Preset(
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
			OkImage.Preset(
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
			OkImage.Preset(
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
			OkImage.Preset(
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
			OkImage.Preset(
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
			PalettizeImage.usePreset( preset )

		print("Demo done!")


	#Map source image unique colors to palette, but avoid collapsing similar colors
	#Return OkImage.UniqueList
	@staticmethod
	def _createWeightedPalette(
		unique_list: OkImage.UniqueList,
		palette_list: OkImage.UniqueList,
		max_error: float = 1.0,
		k_count = 13
	):
		def calcBucketScore(bucket_areas, col_dists, col_idxs, max_radius):
			#lower=better, col dists<1.0, bucket_fullness [0,1]
			area_weight = 10.0 #Arbitrary, bias filled buckets to score worse
			area_weight = max(area_weight,5.0*max_radius) #Allows high max-error to still have an effect
			max_bucket = max(1.0, np.max(bucket_areas))
			area_score = np.maximum(area_weight * bucket_areas[col_idxs]/max_bucket, 1.0)
			return col_dists * area_score

		if unique_list is None:
			raise Exception("src_img Unique_list missing!")
		if palette_list is None:
			raise Exception("palette_img Unique_list missing!")

		pal_length = len(palette_list)

		max_radius = OkTools.approxOkGap(pal_length) * max_error

		#accumulated area of colors in each palette bucket
		bucket_areas = np.zeros(pal_length)

		#Closest palette colors
		pal_tree = palette_list.getUniqueTree()

		packing_density = np.pi / (3.0 * np.sqrt(2.0))
		search_radius = 1.0 + max_error
		est_maxk = packing_density * search_radius**3 + 1.0 #How many palette colors within r=max_error
		k_count = max(2, min(pal_length,est_maxk) )
		dists, idxs = pal_tree.query(unique_list.color, k = int(k_count), workers=-1)
		
		#choose palette index for each color
		unique_count = len(unique_list)
		best_idxs = np.empty(len(unique_list), dtype=int)

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
		
			best_idxs[i] = best_j
			bucket_areas[best_j] += unique_list.area[i]

		unique_mapping = OkImage.UniqueList(
			color=palette_list.color[best_idxs],
			area=unique_list.area.copy(),
			unique_idxs=unique_list.unique_idxs.copy(),
			original_idxs=unique_list.original_idxs.copy()
		)

		return unique_mapping
	


	# Public

	### Palettize Image ###

	@staticmethod
	def usePreset(preset: OkImage.Preset):
		if not preset:
			print("Invalid preset")
			return

		palette_ok = OkImage(preset.palette)
		palette_ok.createUniqueList()
		palette_ok.quantizeAlpha(0)
		pal_length = len(palette_ok.unique_list.color)
		if pal_length < 2:
			print("At least 2 palette colors needed.")
			return

		image_ok = OkImage(preset.image)
		if preset.merge_radius:
			axis_step_size = OkTools.approxOkGap(pal_length) * preset.merge_radius
			axis_count = int(1.0/axis_step_size)
			image_ok.quantizeAxes(axis_count)

		if preset.dither in ["bayer", "blue"]:
			OkImage.Dither.ditherOrdered(image_ok, palette_ok, preset)

		elif preset.dither == "steinberg":
			OkImage.Dither.ditherFloydSteinberg(image_ok, palette_ok, preset)

		elif preset.dither == "none":
			if preset.max_error:
				image_ok.createUniqueList() #only _createWeightedPalette requires source image UniqueList
				#choose closest within max_error weighted by area
				unique_mapping = PalettizeImage._createWeightedPalette(image_ok.unique_list, palette_ok.unique_list, preset.max_error)
				image_ok.applyPalette(unique_mapping)
			else:
				#choose closest to palette
				OkImage.Dither.ditherNone(image_ok, palette_ok, preset)
			#None methods don't affect alpha so we do it here
			image_ok.quantizeAlpha(preset.alpha_count) 
		
		else:
			print("Somehow dither method went missing.")
			exit(-1)

		output_path = os.path.dirname(preset.output)
		if output_path == '':
			output_path = "./"
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		image_ok.saveImage(preset.output)
		print("Saved image "+preset.output)

		if preset.print_stats:
			print("\nColor quant-error")
			image_ok.printImgError()



	## Palettize Image Parser ##


	@staticmethod
	def parser(argv):

		parser = argparse.ArgumentParser(prog=argv[0],description="Palettize and dither using arbitrary palette")

		#all inputs arg_list are strings. Easier to convert str to X
		parser.add_argument(
			'-i', '--input', type=str,
			default= "none",
			help="input .png path"	
		)
		parser.add_argument(
			'-p', '--palette', type=str,
			default= "none",
			help="Palette .png path"	
		)
		parser.add_argument(
			'-o', '--output', type=str,
			default="./",
			help="Output .png path. Empty string \"\", outputs in same dir as input with p_ prefix. No option outputs in current dir."	
		)
		parser.add_argument(
			'-a', '--alpha-count', type=str,
			default="256",
			help="Number of alpha levels. 0 keeps original alpha."
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
			default="bayer",
			help="Options: none, bayer, steinberg, blue"
		)
		parser.add_argument(
			'-ms', '--mask-size', type=str,
			default="16",
			help="Dither matrix size for dither=(bayer, blue)"
		)
		parser.add_argument(
			'-dw', '--mask-weight', type=str,
			default="1.0",
			help="Dither strength for dither=(bayer, blue, steinberg)"
		)
		parser.add_argument(
			'-D', '--demo',  type=str,
			default="False",
			dest='demo',
			help="Generate test images"
		)
		parser.add_argument(
			'-S', '--stats',  type=str,
			default="False",
			dest='print_stats',
			help="Print quant error"
		)

		if len(argv)<=1: #No args
			argv.append("--help")
		arg_list = parser.parse_args(argv[1:])

		#demo has highest priority
		if PalettizeImage._strToBool(arg_list.demo):
			PalettizeImage._demo()
			return None

		#mandatory input files
		if not arg_list.palette or str(arg_list.palette).lower() == "none":
			print("Missing argument -p, --palette <file>")
			return None
		if not arg_list.input or str(arg_list.input).lower() == "none":
			print("Missing argument -i, --input <file>")
			return None

		#alpha count
		if (not arg_list.alpha_count) or str(arg_list.alpha_count).lower() in ["none", ""]:
			arg_list.alpha_count = 0

		#arg_list to preset
		d_preset = OkImage.Preset(
			image				= str(arg_list.input),
			palette			= str(arg_list.palette),
			output			= str(arg_list.output),
			alpha_count		= int(arg_list.alpha_count),
			max_error		= float(arg_list.max_error),
			merge_radius	= float(arg_list.merge_radius),
			dither			= str(arg_list.dither),
			mask_size		= int(arg_list.mask_size),
			mask_weight		= float(arg_list.mask_weight),
			print_stats		= PalettizeImage._strToBool(arg_list.print_stats)
		)

		return d_preset if d_preset.valid else None


