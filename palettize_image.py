#CC0 Kaelygon 2025
"""
Palettize and dither using arbitrary palette
"""

import sys
sys.path.insert(1, './import/')
from PalettizeImage import *

if __name__ == '__main__':
	sys.argv
	argv = sys.argv[:]
 
	im_too_lazy_to_use_terminal = True
	if im_too_lazy_to_use_terminal and len(argv)<=1:
		lazy_arguments = [
			"--demo",			"0", #Generate demo images
			"--input", 			"./demoImages/LPlumocrista.png",
			"--palette", 		"./palettes/pal256.png",
			"--alpha-count", 	"1",		#How many alpha levels
			"--dither", 		"bayer",	#Dither type
			"--bayer-size", 	"0", 	#Bayer matrix size
			"--max-error", 	"0.0",	#Higher will allow farther colros to replace unique colors. Preserves detail but causes banding at high levels. Perfect for pixel art
			"--merge-radius", "0.0",	#Quantize before palettizing. 1.0 will result roughly in same number of colors as palette, but loses information. Reduces unique colors, so 0.05 to 0.2 is good value for high depth images
			"--output",			"./output/test.png",
		]
		for arg in lazy_arguments:
			argv.append(arg)

	d_preset = Palettize_parser(argv)
	if d_preset:
		Palettize_preset(d_preset)
 