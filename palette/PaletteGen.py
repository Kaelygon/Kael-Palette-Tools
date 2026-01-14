import numpy as np

from .PointList import PointList
from .ParticleSim import ParticleSim
from .OkTools import OkTools
from .PointSampler import PointSampler
from .ArrayRandom import ArrayRandom



### Palette generator ###

class PaletteGen:

	@staticmethod
	def _populatePointList(palette_list: PointList, preset):
		"""
			Generate palette where the colors are perceptually evenly spaced out in OKLab colorspace
		"""
		gen_rand = ArrayRandom(preset.seed)

		cell_size = OkTools.approxOkGap(preset.max_colors)
		sample_point_radius = cell_size * preset.sample_radius
		relax_point_radius = cell_size * preset.relax_radius

		if preset.logging:
			print("Using seed ", gen_rand._seed)
			print("Using sample_point_radius "+str(round(sample_point_radius,4)))
			print("Using relax_point_radius "+str(round(relax_point_radius,4)))


		for method in preset.sample_method:
			empty_point_count = preset.max_colors - len(palette_list)
			empty_point_count = max(0,empty_point_count)
			if  empty_point_count<=0:
				break

			new_points = []
			#pre_colors from image
			if method == "precolor":
				new_points = PointSampler.precolor(preset)
			#grayscale points
			elif method == "gray":
				new_points = PointSampler.grayscale(preset.gray_count, sample_point_radius, minimum=OkTools.DARKEST_BLACK_LAB[0])
			#poisson points
			elif method == "poisson":
				new_points = PointSampler.poissonDisk(
					point_list = palette_list,
					point_count = empty_point_count,
					radius = sample_point_radius,
					overlap = 0.01,
					generate_edges = True,
					kissing_number = 48,
					sample_attempts = preset.sample_attempts,
					rand = gen_rand,
					logging = preset.logging
				)
			#random points
			elif method == "random":
				new_points = PointSampler.randomReject(
					point_list = palette_list,
					point_count = empty_point_count,
					radius = sample_point_radius,
					overlap = 0.49,
					sample_attempts = preset.sample_attempts,
					rand = gen_rand,
					logging = preset.logging
				)
			#grid points
			elif method == "grid":
				new_points = PointSampler.gridReject(
					point_list = palette_list,
					point_count = empty_point_count,
					radius = sample_point_radius,
					overlap = 0.3,
					kissing_number = 48,
					sample_attempts = preset.sample_attempts,
					rand = gen_rand,
					logging = preset.logging
				)
			#Handy for testing edge cases
			elif method == "zero":
				new_points = PointSampler.zero(
					point_count = empty_point_count,
					logging = preset.logging
				)

			if len(new_points):
				palette_list.concat(new_points)

		#truncate palette
		palette_list.points = palette_list.points[:preset.max_colors]

		simulator = ParticleSim()
		palette_list = simulator.relaxCloud(
			point_list = palette_list,
			iterations=preset.relax_count,
			rand = gen_rand,
			approx_radius = relax_point_radius,
			record_frame_path = preset.histogram_file,
			log_frequency = preset.logging
		)

		return palette_list


	@staticmethod
	def usePreset(preset):
		palette_list = PointList("oklab")
		palette_list = PaletteGen._populatePointList(palette_list, preset)

		palette_list = PointList.Color.applyLimits(palette_list, preset)
		palette_list = PointList.Color.sort(palette_list, preset)
		output_image = palette_list.saveImage(preset)

		if preset.logging:
			PointList.Stats.gaps(palette_list,precision=4)
			print("Generated "+str(len(palette_list) + preset.reserve_transparent)+" colors to "+ preset.palette_output)
