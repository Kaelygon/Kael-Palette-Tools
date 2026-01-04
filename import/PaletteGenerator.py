import math
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional

import sys
sys.path.insert(1, './import/')

from PointList import *
from oklabConversion import *

from ParticleSim import *
from OkTools import *



### Palette generator ###
@dataclass
class PalettePreset:#
	sample_method: int = 2

	reserve_transparent: bool = True
	hex_pre_colors: List[[str,bool]] = None # ["#0123abc",...]
	img_pre_colors: str = None #file name to existing color palette
	img_fixed_mask: str = None #file name to fixed mask white=fixed black=movable color
	
	max_colors: int = 64		#Max allowed colors including transparency
	gray_count: int = None	#Grayscale color count, None = Auto
	hue_count:  int = 12		#Split Hues in this many buckets

	min_sat: float = 0.0 	#min/max ranges are percentages
	max_sat: float = 1.0
	min_lum: float = 0.0
	max_lum: float = 1.0

	packing_fac: float = 1.0 #Packing efficiency
	max_attempts: int = 1024 #After this many max_attempts per point, point Sampler will give up
	relax_count: int = 64 #number of relax iteration after point sampling
	
	seed: int = 0 # 0=random run to run

	def __post_init__(self):
		if self.packing_fac <= 0:
			self.packing_fac = 1e-12

		if self.max_colors:
			self.max_colors -= self.reserve_transparent #max_count includes transparency
		else:
			self.max_colors = 64



class PointSampler:

	#Simple rejection sampling
	@staticmethod
	def poissonReject(
		point_list: PointList, 
		min_dist: float, 
		point_count: int, 
		max_attempts: int = 1000
	):
		batch_size = 4 * point_count

		output_list = PointList("oklab") 

		attempts=0
		while output_list.length() < point_count:
			attempts+=1
			if attempts > max_attempts:
				break

			np_points = np.random.rand(batch_size, 3) - np.array([0.0, 0.5, 0.5])

			#discard outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			if len(np_points) == 0:
				continue

			#discard if too close
			#query np_points vs (original + previous)
			accepted_points = np.concatenate( [point_list.points["color"], output_list.points["color"]] ) #original points + previous iter
			point_tree = cKDTree(accepted_points)
			dists, _ = point_tree.query(np_points, k=1) 
			not_near = dists >= min_dist
			np_points = np_points[not_near]

			if len(np_points) == 0:
				continue

			#query np_points vs np_points
			point_tree = cKDTree(np_points)
			dists, _ = point_tree.query(np_points, k=2) 
			not_near = dists[:,1] >= min_dist #[:,0] is itself
			np_points = np_points[not_near]

			#concat
			accepted_list = PointList("oklab", len(np_points)) 
			accepted_list.points['color'] = np_points
			accepted_list.points['alpha'] = 1.0
			accepted_list.points['fixed'] = False

			output_list.concat(accepted_list)

		print("PointSampler random() finished after " + str(attempts) + " attempts.")
		output_list.points = output_list.points[:point_count]
		return output_list
		

	#Add points to origin
	@staticmethod
	def zero(preset: PalettePreset, point_count: int):
		oklab_origin = [0.5, 0.0, 0.0]

		accepted_list = PointList("oklab", point_count) 
		accepted_list.points['color'] = oklab_origin
		accepted_list.points['alpha'] = 1.0
		accepted_list.points['fixed'] = False

		return accepted_list


	#Generate grayscale gradient between black and white
	@staticmethod
	def grayscale(point_count: int = None, point_radius: float = None ):
		darkest_black_srgb = [0.499/255,0.499/255,0.499/255] #brighest 8-bit SRGB rounded to pure black 
		darkest_black_lab = srgbToOklab(np.array([darkest_black_srgb]))[0]

		gray_list = PointList("oklab")

		if point_count == None and point_radius:
			point_count = int(round(1.0/(point_radius)))

		if point_count:
			#Use minimum starting luminosity that second darkest black isn't so close to 0
			for i in range(point_count):
				denom = max(1, point_count-1)
				lum = float(i)/((denom))

				#Fade that brightest remains 1.0
				scale = (denom-i)/denom
				lum+= darkest_black_lab[0]*scale

				gray_list.push([lum,0,0], 1.0, True)

		return gray_list


class PaletteGenerator:
	"""
		Generate palette where the colors are perceptually evenly spaced out in OKLab colorspace
	"""

	@staticmethod
	def _applyColorLimits(preset: PalettePreset, point_list: PointList):
		if point_list==None:
			return point_list
		apply_luminosity = preset.max_lum!=1.0 or preset.min_lum!=0.0
		apply_saturation = preset.max_sat!=1.0 or preset.min_sat!=0.0

		max_chroma = math.sqrt(0.5**2+0.5**2)

		not_fixed = ~point_list.points["fixed"]
		color_list = point_list.points["color"][not_fixed]

		if apply_luminosity:
			lum_width = preset.max_lum - preset.min_lum
			color_list[:,0] = color_list[:,0]*lum_width + preset.min_lum
	
		if apply_saturation:
			hued_idxs = ~(OkTools.isOkSrgbGray(color_list))
			hued_colors = color_list[hued_idxs]

			sat_width = preset.max_sat - preset.min_sat
			chroma = OkTools.calcChroma(hued_colors)

			rel_sat = np.zeros_like(chroma)
			rel_sat = chroma/max_chroma
			scaled_sat = (rel_sat * sat_width + preset.min_sat) * max_chroma

			col_vec = hued_colors[:,1:3] #2D Vector a,b
			col_vec = col_vec/chroma[:,None] #Normalize
			col_vec = col_vec*scaled_sat[:,None] #Scale
			color_list[hued_idxs,1:3] = col_vec

		point_list.points["color"][not_fixed] = color_list
		return point_list



	#convert preset.img_pre_colors to PointList. White color in preset.img_fixed_mask makes the point immovable 
	@staticmethod
	def _getPreColors(preset: PalettePreset, alpha_threshold:int=0, fixed_mask_threshold:int=128):
		#Add uint8 colors from image
		pre_palette = Image.open(preset.img_pre_colors)
		pre_palette = pre_palette.convert('RGBA')
		rgba_list = np.array(pre_palette.getdata())
		rgba_list = rgba_list.astype(float) / np.iinfo(np.uint8).max

		lab_list = srgbToOklab(rgba_list[:,:3])
		alpha_list = rgba_list[:,3]

		fixed_list = np.zeros(len(lab_list)) + 1 #if no fixed_mask, default=fixed
		if preset.img_fixed_mask !=None:
			fixed_mask = Image.open(preset.img_fixed_mask)
			fixed_mask = fixed_mask.convert('L')
			fixed_list = np.array(fixed_mask.getdata()) > fixed_mask_threshold

		pre_list = PointList("oklab", len(lab_list)) 
		pre_list.points["color"] = lab_list
		pre_list.points["alpha"] = 1.0
		pre_list.points["fixed"] = fixed_list

		#keep only opaque
		opaque = alpha_list > alpha_threshold
		pre_list.points = pre_list.points[opaque]

		return pre_list


	### Oklab point sampler methods within gamut ###

	#public
	@staticmethod
	def populatePointList(preset : PalettePreset, histogram_path: str = None):	
		cell_size = approxOkGap(preset.max_colors)
		point_radius = cell_size * preset.packing_fac
		print("Using point_radius "+str(round(point_radius,4)))

		palette_list = PointList("oklab")

		#preset points
		if preset.img_pre_colors != None:
			pre_points = PaletteGenerator._getPreColors(preset)
			palette_list.concat(pre_points)

		#grayscale points
		gray_points = PointSampler.grayscale(preset.gray_count, point_radius)
		palette_list.concat(gray_points)

		#poisson points
		empty_point_count = preset.max_colors - palette_list.length()
		empty_point_count = max(0,empty_point_count)
		if preset.sample_method in [0,2] and empty_point_count>0:
			poisson_points = PointSampler.poissonReject( palette_list, point_radius*0.51, empty_point_count )
			palette_list.concat(poisson_points)

		#zero points
		empty_point_count = preset.max_colors - palette_list.length()
		empty_point_count = max(0,empty_point_count)
		if preset.sample_method in [1,2] and empty_point_count>0:
			zero_points = PointSampler.zero(preset, empty_point_count)
			palette_list.concat(zero_points)

		#truncate palette
		palette_list.points = palette_list.points[:preset.max_colors]

		simulator = ParticleSim()
		palette_list = simulator.relaxCloud(
			point_list = palette_list,
			iterations=preset.relax_count,
			approx_radius = point_radius,
			record_frames = histogram_path,
   	)

		palette_list = PaletteGenerator._applyColorLimits(preset, palette_list)

		return palette_list


	@staticmethod
	def saveAsImage(preset: PalettePreset, point_list: PointList, filename: str = "palette.png"):
		rgba = np.zeros((preset.max_colors,4))

		lab_list = point_list.points["color"]
		rgba[:,:3] = oklabToSrgb(lab_list)
		rgba[:,3] = point_list.points["alpha"]

		if preset.reserve_transparent:
			rgba = np.insert(rgba, 0, np.array([0, 0, 0, 0]),axis=0)

		rgba = np.round(rgba*255.0)
		rgba = np.clip(rgba, 0, 255.0)

		arr = np.array([rgba], dtype=np.uint8)
		img = Image.fromarray(arr, mode="RGBA")

		img.save(filename)
		return img

	@staticmethod
	def sortPalette(preset : PalettePreset, point_list : PointList):
		color_list = point_list.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list) 
		gray_colors = color_list[is_gray] 
		hued_colors = color_list[~is_gray]
	
		#bucket similar hues, sort each bucket by luminosity
		hue_bucket_width = 2*math.pi * (1.0/preset.hue_count)

		color_list_hue = np.atan2(hued_colors[:,2], hued_colors[:,1]) + 2* math.pi
		hue_bucket_idxs = color_list_hue/hue_bucket_width
		hue_bucket_idxs = hue_bucket_idxs.astype(int) % preset.hue_count #hue_bucket_idxs[i] = hue_idx of color_list[i]

		sorted_gray_colors = gray_colors[np.argsort(gray_colors[:, 0])]
		sorted_colors = sorted_gray_colors
		for idx in range(preset.hue_count):
			hue_bucket = hued_colors[ np.where(hue_bucket_idxs==idx) ] #bucket colors with same hue_idx
			hue_bucket = hue_bucket[np.argsort(hue_bucket[:, 0])] #sort bucket by luminosity
			sorted_colors = np.concatenate([sorted_colors, hue_bucket])

		point_list.points["color"] = sorted_colors
		return point_list

	#EOF PaletteGenerator

