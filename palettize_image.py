#!/usr/bin/env python
#CC0 Kaelygon 2025
"""
Palettize and dither using arbitrary palette
"""

import sys
from palette.PalettizeImage import *


if __name__ == '__main__':
	sys.argv
	argv = sys.argv[:]

	im_too_lazy_to_use_terminal = False
	if im_too_lazy_to_use_terminal and len(argv)<=1:
		lazy_arguments = [
			"--demo",			"False", #Generate demo images
			"--input", 			"./demoImages/LPlumocrista.png",
			"--palette", 		"./palettes/pal256.png",
			"--alpha-count", 	"256",	#How many alpha levels
			"--dither", 		"blue",	#Dither type
			"--max-error", 	"0.0",	#Higher will allow farther colors to replace unique colors. Preserves detail
			"--merge-radius", "0.0",	#Quantize before palettizing. 0.05 to 0.2 is good value for high depth images
 			"--mask-size", 	"128", 	#dither matrix size
			"--mask-weight",	"1.0",	#Dither strength for dither = blue or bayer
			"--stats",			"False",
			"--output",			"./output/lazy_test.png",
		]
		for arg in lazy_arguments:
			argv.append(arg)

	d_preset = PalettizeImage.parser(argv)
	if d_preset:
		PalettizeImage.usePreset(d_preset)
