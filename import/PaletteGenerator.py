#PaletteGenerator.py
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from dataclasses import dataclass, field

import sys
sys.path.insert(1, './import/')

from PointList import *

from ParticleSim import *
from OkTools import *

from PalettePreset import *
from PointSampler import *




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

		max_chroma = np.sqrt(0.5**2+0.5**2)

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


	### Oklab point sampler methods within gamut ###

	#public
	@staticmethod
	def populatePointList(preset : PalettePreset):
		if preset.seed == None:
			preset.seed = int(np.random.SeedSequence().entropy)
			rand = np.random.default_rng(preset.seed)
		elif preset.seed == 0:
			rand = None #No generator
		else:
			rand = np.random.default_rng(preset.seed)
		

		cell_size = approxOkGap(preset.max_colors)
		sample_point_radius = cell_size * preset.sample_radius
		relax_point_radius = cell_size * preset.relax_radius

		if preset.logging:
			print("Using seed: " + str(preset.seed))
			print("Using sample_point_radius "+str(round(sample_point_radius,4)))
			print("Using relax_point_radius "+str(round(relax_point_radius,4)))

		palette_list = PointList("oklab",0)

		for method in preset.sample_method:
			empty_point_count = preset.max_colors - len(palette_list)
			empty_point_count = max(0,empty_point_count)
			if  empty_point_count<=0:
				break

			#pre_colors from image
			elif method == "precolor":
				new_points = PointSampler.precolor(preset)

			#grayscale points
			elif method == "gray":
				new_points = PointSampler.grayscale(preset.gray_count, sample_point_radius, minimum=OkTools.DARKEST_BLACK_LAB[0])

			#poisson points
			elif method == "poisson":
				new_points = PointSampler.poissonDisk(
					rand = rand,
					point_list=palette_list, 
					point_count=empty_point_count, 
					radius=sample_point_radius, 
					overlap = 0.0,
					generate_edges = True,
					kissing_number = 48,
					sample_attempts = preset.sample_attempts,
					logging = preset.logging
				)

			#random points
			elif method == "random":
				new_points = PointSampler.randomReject( 
					rand=rand, 
					point_list = palette_list, 
					point_count = empty_point_count, 
					radius=sample_point_radius, 
					overlap = 0.49,
					sample_attempts = preset.sample_attempts,
					logging = preset.logging
				)

			#grid points
			elif method == "grid":
				new_points = PointSampler.gridReject( 
					point_list = palette_list, 
					point_count = empty_point_count, 
					radius=sample_point_radius, 
					overlap = 0.3,
					kissing_number = 48,
					sample_attempts = preset.sample_attempts,
					logging = preset.logging
				)

			#Handy for testing edge cases
			elif method == "zero":
				new_points = PointSampler.zero(preset, empty_point_count)

			if len(new_points):
				palette_list.concat(new_points)


		#truncate palette
		palette_list.points = palette_list.points[:preset.max_colors]

		simulator = ParticleSim()
		palette_list = simulator.relaxCloud(
			point_list = palette_list,
			iterations=preset.relax_count,
			approx_radius = relax_point_radius,
			record_frame_path = preset.histogram_file,
			rand = rand,
			log_frequency = preset.logging * 64
   	)

		palette_list = PaletteGenerator._applyColorLimits(preset, palette_list)

		return preset, palette_list


	@staticmethod
	def saveAsImage(preset: PalettePreset, point_list: PointList):
		p_count = len(point_list)
		if(len(point_list)==0 and preset.reserve_transparent == 0):
			return

		rgba = np.zeros((p_count,4))

		lab_list = point_list.points["color"]
		rgba[:,:3] = oklabToSrgb(lab_list)
		rgba[:,3] = point_list.points["alpha"]

		if preset.reserve_transparent:
			rgba = np.insert(rgba, 0, np.array([0, 0, 0, 0]),axis=0)

		rgba = np.round(rgba*255.0)
		rgba = np.clip(rgba, 0, 255.0)

		arr = np.array([rgba], dtype=np.uint8)
		img = Image.fromarray(arr, mode="RGBA")

		img.save(preset.palette_output)
		return img

	@staticmethod
	def sortPalette(preset : PalettePreset, point_list : PointList):
		color_list = point_list.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list) 
	
		#bucket similar hues, sort each bucket by luminosity
		hue_bucket_width = 2*np.pi * (1.0/preset.hue_count)

		color_list_hue = np.atan2(color_list[:,2], color_list[:,1]) + 2* np.pi
		hue_bucket_idxs = color_list_hue/hue_bucket_width
		hue_bucket_idxs = hue_bucket_idxs.astype(int) % preset.hue_count #hue_bucket_idxs[i] = hue_idx of color_list[i]
		hue_bucket_idxs[is_gray] = -1

		sorted_idxs = np.array([],dtype=int)
		for idx in range(-1, preset.hue_count):
			this_bucket_idxs = np.where(hue_bucket_idxs==idx)[0] #bucket colors with same hue_idx
			bucket_colors = color_list[this_bucket_idxs]
			sorted_sub_idxs= np.argsort(bucket_colors[:, 0]) #sort bucket by luminosity
			this_bucket_sorted_idxs = this_bucket_idxs[sorted_sub_idxs]

			sorted_idxs = np.concatenate([sorted_idxs, this_bucket_sorted_idxs])

		point_list.points = point_list.points[sorted_idxs]
		sorted_idxs = np.sort(sorted_idxs)
		return point_list

	#EOF PaletteGenerator

