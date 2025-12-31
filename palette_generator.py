
"""
	Generate uniform palette in OKLab gamut
"""

import os
import sys
sys.path.insert(1, './import/')
from PaletteGenerator import *

def run_PaletteGenerator(active_preset, palette_file, histogram_file):
	palette = PaletteGenerator(preset=active_preset)
	palette.populatePointCloud(histogram_file)

	palette.sortPalette()

	palette.paletteToImg(palette_file)

	PointGridStats.printGapStats(palette.point_grid,4)

	hex_string = palette.paletteToHex()	
	printHexList(hex_string,palette_name)

	print("Generated "+str(len(palette.point_grid.cloud) + active_preset.reserve_transparent)+" colors to "+palette_file)


if __name__ == '__main__':
	preset_pal64 = PalettePreset(
			sample_method=2,
			reserve_transparent=1,
			hex_pre_colors = None, #[["abcdef",True],["abc123ff",False]],
			img_pre_colors = "output/pal64-base.png",#"./data/pre_palette.png",
			img_fixed_mask = "output/pal64-fixed.png", #"./data/pre_palette_mask.png",

			gray_count	=15, 
			max_colors	=256, 
			hue_count	=12,
			min_sat		=0.0, 
			max_sat		=1.0, 
			min_lum		=0.0, 
			max_lum		=1.0,
		
			packing_fac	=1.2,
			max_attempts=1024*32,
			relax_count =512,
			seed=0
		)

	output_path = "./output"
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	palette_name = 'testPalette.png'
	histogram_file = output_path + "/" + "cloudHistogram.py"
	palette_file = output_path + "/" + palette_name

	run_PaletteGenerator(preset_pal64, palette_file, histogram_file)