#palette_generator.py
"""
	Generate uniform palette in OKLab gamut
"""

import os
import sys
import numpy as np
sys.path.insert(1, './import/')
from PaletteGenerator import *
from PointListStats import *
from PalettePreset import *

def run_PaletteGenerator(preset):
	palette_list = PointList("oklab")
	preset, palette_list = PaletteGenerator.populatePointList(preset)
	palette_list = PaletteGenerator.sortPalette(preset, palette_list)

	PaletteGenerator.saveAsImage(preset, palette_list)

	if preset.logging:
		PointListStats.printGapStats(palette_list,4)
		print("Generated "+str(len(palette_list) + preset.reserve_transparent)+" colors to "+ preset.palette_output)


if __name__ == '__main__':
	output_path = "./output/"
	preset_pal64 = PalettePreset(
			sample_method = ["gray", "poisson", "grid"],
			reserve_transparent=1,
			img_pre_colors = "palettes/pal64.png",
			img_fixed_mask = "palettes/pal64-fixed.png",
			histogram_file = output_path + "cloudHistogram.npy",
			palette_output = output_path + "testPalette.png",

			gray_count	=6, 
			max_colors	=64, 
			hue_count	=12,
			min_sat		=0.0, 
			max_sat		=1.0, 
			min_lum		=0.0, 
			max_lum		=1.0,
		
			sample_radius = 1.2,
			sample_attempts = 1024,

			relax_radius = 1.3,
			relax_count = 1024,
			seed=0,

			logging=True
		)

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	run_PaletteGenerator(preset_pal64)
