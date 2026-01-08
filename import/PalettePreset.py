#PalettePreset.py
from dataclasses import dataclass, field
import numpy as np
import os.path

### Palette generator ###
@dataclass
class PalettePreset:
	"""Preset for PaletteGenerator"""

	#File i/o
	palette_output: str = "palette.png" # (mandatory) full output file path
	img_pre_colors: str = None # (optional) file name to existing color palette
	img_fixed_mask: str = None # (optional) file name to fixed mask white=fixed black=movable color
	histogram_file: str = None # (optional) particle sim point frame history for ok_plot_histogram.py

	#sample_method can be any combination of these
	#"precolor" obtain colors from img_pre_colors, pixels in img_fixed_mask with luminance>128 retain their color  
	#"gray" generate gray_count grayscale colors 
	#"poisson" poisson disk sample
	#"voids" find and fill voids 
	#"random" random reject, allows 51% sample_radius overlap, requires seed>0
	#"zero" generate points at [0.5,0.0,0.0]
	sample_method: list[str] = field(default_factory=lambda: ["gray", "poisson", "grid"]) 

	reserve_transparent: int = 1

	gray_count: int = None	#Grayscale color count, None = Auto
	max_colors: int = 64		#Max allowed colors including transparency
	hue_count:  int = 12		#Split Hues in this many buckets

	min_sat: float = 0.0 	#min/max ranges are percentages
	max_sat: float = 1.0
	min_lum: float = 0.0
	max_lum: float = 1.0

	sample_radius: float = 1.0 # radius used in PointSampler 
	relax_radius: float = 1.2 # radius used in ParticleSim

	sample_attempts: int = 1024 #After this many sample_attempts per point, sampler method will give up
	relax_count: int = 1024 #number of relax iteration after point sampling
	
	seed: int = None #None = random seed, 0 = no generator, [1,UINT64_MAX] = set seeded

	logging: bool = False #Disables stats and some printing

	def __post_init__(self):
		self.VALID_METHODS: list[str] = ["precolor", "gray", "poisson", "grid", "random", "zero"]
		self.DEFAULT_METHOD: list[str] = ["gray", "poisson","grid"]

		#var sanity checks
		self.reserve_transparent = max(0, min(1, self.reserve_transparent) )

		if self.sample_radius <= 0:
			self.sample_radius = 1e-12

		if self.max_colors:
			self.max_colors -= self.reserve_transparent #max_count includes transparency
		else:
			self.max_colors = 64

		#Sample method
		has_valid_method = False
		for i,method in enumerate(self.sample_method):
			self.sample_method[i] = method.lower()
			if method not in self.VALID_METHODS:
				print("invalid method ", method)
			else:
				has_valid_method = True

		if has_valid_method == False:
			self.sample_method = self.DEFAULT_METHOD
			print("No valid sample_method. Defaulting to ", str(self.DEFAULT_METHOD))

		#input validity check
		file_attributes = ["palette_output", "histogram_file", "img_pre_colors", "img_fixed_mask"]
		is_input = [0, 0, 1, 1]
		for i,attribute in enumerate(file_attributes):
			file = getattr(self, attribute)
			if file == None:
				continue

			base_dir = os.path.dirname(file)
			if base_dir == '':
				base_dir = "./"

			preset_fail = 0

			#All file checks
			if not os.path.isdir(base_dir):
				print("Directory "+base_dir+" doesn't exist.")
				preset_fail = 1

			if is_input[i]:
				#input file checks
				if not os.path.exists(file) or not os.access(base_dir, os.R_OK):
					print("File doesn't exist "+file)
					preset_fail = 1

			else:
				#output file checks
				if not os.access(base_dir, os.W_OK):
					print("Can't access directory "+base_dir)
					preset_fail = 1

			if preset_fail:
				setattr(self, attribute, None)
