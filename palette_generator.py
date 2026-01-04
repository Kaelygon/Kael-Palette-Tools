
"""
	Generate uniform palette in OKLab gamut
"""

import os
import sys
import numpy as np
sys.path.insert(1, './import/')
from PaletteGenerator import *
from PointListStats import *

def run_PaletteGenerator(preset : PalettePreset, output_file, histogram_file):
	np.random.seed(preset.seed)

	point_list = PaletteGenerator.populatePointList(preset, histogram_file)
	point_list = PaletteGenerator.sortPalette(preset, point_list)

	PaletteGenerator.saveAsImage(preset, point_list, output_file)

	PointListStats.printGapStats(point_list,4)
	print("Generated "+str(point_list.length() + preset.reserve_transparent)+" colors to "+output_file)


if __name__ == '__main__':
	preset_pal64 = PalettePreset(
			sample_method=2,
			reserve_transparent=1,
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
			relax_count =256,
			seed=0
		)

	output_path = "./output"
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	palette_name = 'testPalette.png'
	histogram_file = output_path + "/" + "cloudHistogram.py"
	palette_file = output_path + "/" + palette_name

	run_PaletteGenerator(preset_pal64, palette_file, histogram_file)