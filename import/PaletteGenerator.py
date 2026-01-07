#PaletteGenerator.py
import math
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from dataclasses import dataclass, field

import sys
sys.path.insert(1, './import/')

from PointList import *

from ParticleSim import *
from OkTools import *



### Palette generator ###
@dataclass
class PalettePreset:
	sample_method: int = 2 #0 = poisson_reject, 1 = random, 2 = poisson_reject + fallback random

	reserve_transparent: int = 1
	img_pre_colors: str = None #file name to existing color palette
	img_fixed_mask: str = None #file name to fixed mask white=fixed black=movable color
	
	max_colors: int = 64		#Max allowed colors including transparency
	gray_count: int = None	#Grayscale color count, None = Auto
	hue_count:  int = 12		#Split Hues in this many buckets

	min_sat: float = 0.0 	#min/max ranges are percentages
	max_sat: float = 1.0
	min_lum: float = 0.0
	max_lum: float = 1.0

	sample_radius: float = 0.9 # radius used in PointSampler 
	relax_radius: float = 1.2, # radius used in ParticleSim

	sample_attempts: int = 1024 #After this many sample_attempts per point, point Sampler will give up
	relax_count: int = 64 #number of relax iteration after point sampling
	
	seed: int = None

	def __post_init__(self):
		self.reserve_transparent = max(0, min(1, self.reserve_transparent) )

		if self.sample_radius <= 0:
			self.sample_radius = 1e-12

		if self.max_colors:
			self.max_colors -= self.reserve_transparent #max_count includes transparency
		else:
			self.max_colors = 64



class PointSampler:
	#query np_points vs other_points, remove points np_points that are closer than min_dist to other_points
	@staticmethod
	def _removeNearbyPoints(np_points, other_points, min_dist):
		if len(np_points) == 0 or len(other_points) == 0:
			return np_points
		point_tree = cKDTree(other_points)
		dists, _ = point_tree.query(np_points, k=1) 
		not_near = dists >= min_dist
		np_points = np_points[not_near]
		return np_points

	#query np_points vs np_points, remove other point that's closer than min_dist
	@staticmethod
	def _removeNearbySelf(np_points, min_dist):
		if len(np_points) < 2:
			return np_points
		point_tree = cKDTree(np_points)
		pairs = point_tree.query_pairs(min_dist)
		#remove only one of the pair points that is too close
		if pairs:
			pairs = np.array(list(pairs))
			too_close = pairs[:,0]
			np_points = np.delete(np_points, too_close, axis=0)
		return np_points

	#Generate surface normals of n points evenly distributed on a sphere 
	@staticmethod
	def _sphere_normals(normals_count):
		indices = np.arange(0, normals_count, dtype=float) + 0.5
		phi = np.arccos(1 - 2*indices/normals_count)
		theta = np.pi * (1 + 5**0.5) * indices

		x = np.sin(phi) * np.cos(theta)
		y = np.sin(phi) * np.sin(theta)
		z = np.cos(phi)

		return np.stack((x, y, z), axis=1)


	@staticmethod
	def poissonDisk(
		rand: np.random.Generator,
		point_list: PointList,
		point_count: int,
		radius: float,
		overlap: float = 0.0,
		kissing_number = 48
	):
		initial_points = point_list.points["color"]
		min_dist = radius - radius * overlap

		offset_list = PointSampler._sphere_normals(kissing_number) * radius

		#Generate corners as starting seed
		xx,yy,zz = np.meshgrid([0,1], [0,1], [0,1], indexing='ij')
		linear_corners = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
		ok_corners = linearToOklab(linear_corners)

		#remove corners close to initial_points
		ok_corners = PointSampler._removeNearbyPoints(ok_corners, initial_points, min_dist)

		accepted_points = np.empty((0,3))
		accepted_points = np.vstack((accepted_points, ok_corners))
		active_points = np.vstack((accepted_points, initial_points))


		iterations = 0
		while accepted_points.shape[0] < point_count and active_points.shape[0]:
			iterations+=1
			#Generate sphere of points around each point
			np_points = active_points[:, None, :] + offset_list[None, :, :]
			np_points = np_points.reshape(-1, 3)

			#remvoe outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			#remove nearby points
			all_points = np.vstack((accepted_points, initial_points))
			np_points = PointSampler._removeNearbyPoints(np_points, all_points, min_dist)
			np_points = PointSampler._removeNearbySelf(np_points, min_dist)

			#new points become new seeds
			accepted_points = np.vstack((accepted_points, np_points))
			active_points = np_points

		#to PointList
		accepted_points = accepted_points[:point_count]
		if len(accepted_points):
			output_list = PointList("oklab", len(accepted_points))
			output_list.points['color'] = accepted_points
			output_list.points['alpha'] = 1.0
			output_list.points['fixed'] = False
		else:
			return PointList("oklab", 0)

		print("PointSampler poissonDisk() finished after " + str(iterations) + " iterations.")
		return output_list


	#Simple rejection sampling
	@staticmethod
	def randomReject(
		rand: np.random.Generator,
		point_list: PointList, 
		point_count: int, 
		radius: float = None, 
		overlap: float = None, #None skips distance check
		sample_attempts: int = 1000
	):
		box_volume = np.prod(OkTools.OKLAB_BOX_SIZE)
		in_gamut_prob = OkTools.OKLAB_GAMUT_VOLUME / box_volume #probability being in gamut

		min_dist = None
		batch_mul = 1.0 
		if overlap!=None and radius != None: 
			batch_mul = 1.0 / overlap #guesstimate point_count + reject_count of random point_cloud
			overlap = min(max(overlap, 0.05), 0.95)
			min_dist = radius - radius * overlap

		batch_size = int(batch_mul * point_count / in_gamut_prob) +1

		initial_points = point_list.points["color"]
		accepted_points = np.empty((0,3)) 

		attempts=0
		while len(accepted_points) < point_count:
			attempts+=1
			if attempts > sample_attempts:
				break

			np_points = rand.random((batch_size, 3))*OkTools.OKLAB_BOX_SIZE + OkTools.OKLAB_BOX_MIN

			#discard outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			if len(np_points) == 0:
				continue

			#discard if too close
			if min_dist != None:
				all_points = np.vstack((accepted_points, initial_points))
				np_points = PointSampler._removeNearbyPoints(np_points, accepted_points, min_dist)
				np_points = PointSampler._removeNearbySelf(np_points, min_dist)

			accepted_points = np.vstack((accepted_points, np_points))

		#to PointList
		accepted_points = accepted_points[:point_count]
		if len(accepted_points):
			output_list = PointList("oklab", len(accepted_points))
			output_list.points['color'] = accepted_points
			output_list.points['alpha'] = 1.0
			output_list.points['fixed'] = False
		else:
			return PointList("oklab", 0)

		print("PointSampler random() finished after " + str(attempts) + " attempts.")
		return output_list
		

	#Add points to origin
	@staticmethod
	def zero(preset: PalettePreset, point_count: int):
		oklab_origin = [0.5, 0.0, 0.0]

		accepted_points = PointList("oklab", point_count) 
		accepted_points.points['color'] = oklab_origin
		accepted_points.points['alpha'] = 1.0
		accepted_points.points['fixed'] = False

		return accepted_points


	#Generate grayscale gradient between black and white
	@staticmethod
	def grayscale(point_count: int = None, point_radius: float = None ):

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
				lum+= OkTools.DARKEST_BLACK_LAB[0]*scale

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
	def populatePointList(preset : PalettePreset, palette_list: PointList, histogram_path: str = None):
		if preset.seed == None:
			preset.seed = np.random.SeedSequence().entropy
		rand = np.random.default_rng(preset.seed)
		
		print("Using seed: " + str(preset.seed))

		cell_size = approxOkGap(preset.max_colors)
		sample_point_radius = cell_size * preset.sample_radius
		relax_point_radius = cell_size * preset.relax_radius
		print("Using sample_point_radius "+str(round(sample_point_radius,4)))
		print("Using relax_point_radius "+str(round(relax_point_radius,4)))

		#preset points
		if preset.img_pre_colors != None:
			pre_points = PaletteGenerator._getPreColors(preset)
			palette_list.concat(pre_points)

		#grayscale points
		gray_points = PointSampler.grayscale(preset.gray_count, sample_point_radius)
		palette_list.concat(gray_points)

		#poisson points
		empty_point_count = preset.max_colors - len(palette_list)
		empty_point_count = max(0,empty_point_count)
		if preset.sample_method in [0,2] and empty_point_count>0:
			poisson_points = PointSampler.poissonDisk(
				rand = rand,
				point_list=palette_list, 
				point_count=empty_point_count, 
				radius=sample_point_radius, 
				overlap = 0.0
			)
			palette_list.concat(poisson_points)

		#fallback random
		empty_point_count = preset.max_colors - len(palette_list)
		empty_point_count = max(0,empty_point_count)
		if preset.sample_method in [1,2] and empty_point_count>0:
			random_points = PointSampler.randomReject( 
				rand=rand, 
				point_list = palette_list, 
				point_count = empty_point_count, 
				radius=sample_point_radius, 
				overlap = 0.49,
				sample_attempts = preset.sample_attempts
			)
			palette_list.concat(random_points)

		#truncate palette
		palette_list.points = palette_list.points[:preset.max_colors]

		simulator = ParticleSim()
		palette_list = simulator.relaxCloud(
			point_list = palette_list,
			iterations=preset.relax_count,
			approx_radius = relax_point_radius,
			record_frame_path = histogram_path,
			rand = rand
   	)

		palette_list = PaletteGenerator._applyColorLimits(preset, palette_list)

		return preset, palette_list


	@staticmethod
	def saveAsImage(preset: PalettePreset, point_list: PointList, filename: str = "palette.png"):
		p_count = len(point_list)
		if(len(point_list)==0):
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

		img.save(filename)
		return img

	@staticmethod
	def sortPalette(preset : PalettePreset, point_list : PointList):
		color_list = point_list.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list) 
	
		#bucket similar hues, sort each bucket by luminosity
		hue_bucket_width = 2*math.pi * (1.0/preset.hue_count)

		color_list_hue = np.atan2(color_list[:,2], color_list[:,1]) + 2* math.pi
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

