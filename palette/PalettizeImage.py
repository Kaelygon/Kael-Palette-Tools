"""Run OkImage.Preset through OkImage  """
import numpy as np
from scipy.spatial import cKDTree

import os.path
import argparse

from .OkImage import OkImage
from .OkTools import OkTools



class PalettizeImage:
	"""Interpret sys.argv to create palettized images"""

	#CLI -> parser -> return preset or run demo
	#
	#usePreset -> load and convert images to OkImage 
	#-> run filters set by preset -> save images

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
				image				= os.path.join(input_path,"KaelygonLogo25.png"),
				palette			= os.path.join(palette_path,"pal16.png"),
				output			= os.path.join(output_path,"p_KaelygonLogo25.png"),
				max_error		= 2.0,
				merge_radius	= 0.0,
				dither			= "none",
			),
			OkImage.Preset(
				image				= os.path.join(input_path,"LPlumocrista.png"),
				palette			= os.path.join(palette_path,"pal256.png"),
				output			= os.path.join(output_path,"p_LPlumocrista.png"),
				dither			= "bayer",
				mask_size		= 16,
				mask_weight		= 1.0,
			),
			OkImage.Preset(
				image				= os.path.join(input_path,"Michelangelo_David.png"),
				palette			= os.path.join(palette_path,"wplacePalette.png"),
				output			= os.path.join(output_path,"p_Michelangelo_David.png"),
				max_error		= 1.0,
				merge_radius	= 0.1,
				dither			= "steinberg",
			),
			OkImage.Preset(
				image				= os.path.join(input_path,"gray.png"),
				palette			= os.path.join(palette_path,"pal2.png"),
				output			= os.path.join(output_path,"p_gray.png"),
				dither			= "bayer",
				mask_size		= 16,
			),
			OkImage.Preset(
				image				= os.path.join(input_path,"rgba24_color_test.png"),
				palette			= os.path.join(palette_path,"pal64.png"),
				output			= os.path.join(output_path,"p_rgba24_color_test.png"),
				alpha_count		= 16,
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


	#ATM This is the only mapping function so this sits in PalettizeImage namespace
	@staticmethod
	def _createWeightedPalette(
		unique_list: OkImage.UniqueList,
		palette_list: OkImage.UniqueList,
		max_error: float = 1.0,
		k_count = 13
	):
		"""
			Map source image unique colors to palette, but avoid collapsing similar colors
			Return OkImage.UniqueList
		"""

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
		pal_tree = palette_list.tree

		packing_density = np.pi / (3.0 * np.sqrt(2.0))
		search_radius = 1.0 + max_error
		est_maxk = packing_density * search_radius**3 + 1.0 #How many palette colors within r=max_error
		k_count = max(2, min(pal_length,est_maxk) )
		dists, idxs = pal_tree.query(unique_list.color, k = int(k_count), workers=-1)
		
		#choose palette index for each color
		unique_count = len(unique_list)
		best_idxs = np.empty(len(unique_list), dtype=int)

		#prioritize largest area
		unique_sorted_idx = np.argsort(-1.0*unique_list.area, kind='stable')
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

		#UniqueList used as mapping
		unique_mapping = unique_list.copy()
		unique_mapping.color = palette_list.color[best_idxs]
		return unique_mapping
	


	#calc stats of two image difference
	@staticmethod
	def _printImgError(img1, img2):
		quant_delta = img2.pixels - img1.pixels
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




	# Public

	### Palettize Image ###

	@staticmethod
	def usePreset(preset: OkImage.Preset):
		if not preset:
			print("Invalid preset")
			return

		image_ok = OkImage()
		image_ok.loadImage(preset.image)

		if preset.print_stats:
			#save copy for stats
			original_ok = image_ok.copy()

		palette_ok = OkImage()
		palette_ok.loadImage(preset.palette)
		OkImage.Filter.quantizeAxes(palette_ok, step_count=1, axes=[3]) #remove palette alpha
		palette_ok.updateUniqueList()

		pal_length = len(palette_ok.unique_list.color)
		if pal_length < 2:
			print("At least 2 palette colors needed.")
			return

		#First quantize pass to reduce colors user defined depth
		if preset.merge_radius:
			axis_step_size = OkTools.approxOkGap(pal_length) * preset.merge_radius
			axis_count = int(1.0/axis_step_size)
			OkImage.Filter.quantizeAxes(palette_ok, step_count=axis_count, axes=[0,1,2])

		#Apply dither methods
		if preset.dither in ["bayer", "blue"]:
			OkImage.Filter.ditherOrdered(image_ok, palette_ok, preset)

		elif preset.dither == "steinberg":
			OkImage.Filter.ditherFloydSteinberg(image_ok, palette_ok, preset)

		elif preset.dither == "none":
			if preset.max_error:
				#choose closest within max_error, weighted by area
				image_ok.updateUniqueList() #only place where source image uniques are calculated
				unique_mapping = PalettizeImage._createWeightedPalette(image_ok.unique_list, palette_ok.unique_list, preset.max_error)
				OkImage.Filter.applyPalette(image_ok, unique_mapping)
			else:
				#choose closest to palette
				OkImage.Filter.ditherNone(image_ok, palette_ok)
			#dither="none" methods don't affect alpha so we do it here
			OkImage.Filter.quantizeAxes(image_ok, step_count=preset.alpha_count, axes=[3]) 
		
		else:
			raise Exception("Somehow dither method went missing.")

		output_path = os.path.dirname(preset.output)
		if output_path == '':
			output_path = "./"
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		image_ok.saveImage(preset.output)
		print("Saved image "+preset.output)

		if preset.print_stats:
			print("\nColor quant-error")
			PalettizeImage._printImgError(original_ok, image_ok)



	## Palettize Image Parser ##


	@staticmethod
	def parser(argv):
		"""Convert and assign sys.argv to OkImage.Preset"""

		parser = argparse.ArgumentParser(prog=argv[0],description="Palettize and dither using arbitrary palette")

		#all inputs arg_list are strings. Easier to convert str to X
		parser.add_argument(
			'-i', '--input', type=str,
			default= "",
			help="input .png path"	
		)
		parser.add_argument(
			'-p', '--palette', type=str,
			default= "",
			help="Palette .png path"	
		)
		parser.add_argument(
			'-o', '--output', type=str,
			default="./",
			help="Output .png path. Empty string \"\", outputs in same dir as input with p_ prefix. No option outputs in current dir."	
		)
		parser.add_argument(
			'-a', '--alpha-count', type=str,
			default="",
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
			help="Filter matrix size for dither=(bayer, blue)"
		)
		parser.add_argument(
			'-dw', '--mask-weight', type=str,
			default="1.0",
			help="Filter strength for dither=(bayer, blue, steinberg)"
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
		if not arg_list.palette or str(arg_list.palette).lower() == "":
			print("Missing argument -p, --palette <file>")
			return None
		if not arg_list.input or str(arg_list.input).lower() == "":
			print("Missing argument -i, --input <file>")
			return None

		#alpha count
		if (not arg_list.alpha_count) or str(arg_list.alpha_count).lower() == "":
			arg_list.alpha_count = 0

		#Convert strings to valid types
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


