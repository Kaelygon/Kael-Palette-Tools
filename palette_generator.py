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

def run_PaletteGenerator(preset, output_file, histogram_file):
	"""void run_PaletteGenerator(PalettePreset preset, string output_file, string histogram_file)""" 

	palette_list = PointList("oklab")
	preset, palette_list = PaletteGenerator.populatePointList(preset, palette_list, histogram_file)
	palette_list = PaletteGenerator.sortPalette(preset, palette_list)

	PaletteGenerator.saveAsImage(preset, palette_list, output_file)

	PointListStats.printGapStats(palette_list,4)
	print("Generated "+str(len(palette_list) + preset.reserve_transparent)+" colors to "+output_file)


if __name__ == '__main__':
	preset_pal64 = PalettePreset(
			sample_method=0,
			reserve_transparent=0,
			img_pre_colors = None, #"output/pal64-base.png",
			img_fixed_mask = None, #"output/pal64-fixed.png",

			gray_count	=6, 
			max_colors	=256, 
			hue_count	=12,
			min_sat		=0.0, 
			max_sat		=1.0, 
			min_lum		=0.0, 
			max_lum		=1.0,
		
			packing_fac	=1.2,
			max_attempts=1024,
			relax_count =1024,
			seed=0
		)

	output_path = "./output"
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	palette_name = 'testPalette.png'
	histogram_file = output_path + "/" + "cloudHistogram"
	palette_file = output_path + "/" + palette_name

	run_PaletteGenerator(preset_pal64, palette_file, histogram_file)
