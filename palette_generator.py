#!/usr/bin/env python
"""
	Generate uniform palette in OKLab gamut
"""

import os
from palette import PaletteGen, PalettePreset


if __name__ == '__main__':
	output_path = "output"
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	preset_pal64 = PalettePreset(
			sample_method = ["gray", "poisson", "grid", "random", "zero"],
			reserve_transparent=1,
			img_pre_colors = "palettes/pal64.png", #sample_method must include "precolor"
			img_fixed_mask = "palettes/pal64-fixed.png", #white = fix img_pre_colors, black = movable

			histogram_file = os.path.join(output_path, "cloudHistogram.npy"), #relax sim point positions (optional)
			palette_output = os.path.join(output_path, "testPalette.png"),

			gray_count	= None, #None=Auto, 0=no gray. sample_method must include "gray"
			max_colors	= 255, #excludes transparency
			hue_count	= 12, #sort output palette into this many hues

			min_sat		= 0.0, #postprocess palette
			max_sat		= 1.0,
			min_lum		= 0.0,
			max_lum		= 1.0,
		
			sample_radius = 1.1, #point_sampler target point separation
			sample_attempts = 1024,

			relax_radius = 1.3, #simulation target point separation
			relax_count = 1024, #relax iterations
			seed = None, #point sampler and relax random seed, None = random

			logging = 256 #0=disable, 1 enables general logging. Additionally: >1 logs each time this many relax ticks pass.
		)

	if preset_pal64.valid:
		PaletteGen.usePreset(preset_pal64)
	else:
		print("Invalid preset")