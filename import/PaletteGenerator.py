##CC0 Kaelygon 2025
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional

import sys
sys.path.insert(1, './import/')

from float3 import *
from RorrLcg import *
from KaelColor import *
from PointGrid import *
from ParticleSim import *
from PointGridStats import *


"""
Numpy was too slow because how the points are handled so we using float3
"""

### misc tools
def hexToSrgba(hexstr: str):
	s = hexstr.strip().lstrip('#')
	if len(s) < 6:
		return [0,0,0,0]
	r = int(s[0:2], 16) / 255.0
	g = int(s[2:4], 16) / 255.0
	b = int(s[4:6], 16) / 255.0
	a = 1.0
	if len(s) == 8:
		a = int(s[6:8], 16) / 255.0
	return [r, g, b, a]

def printHexList(hex_list, palette_name="", new_line=True):
	out_string= palette_name + " = "
	out_string+="["
	for string in hex_list:
		out_string += "\""+string+"\","
	out_string+="]"
	print(out_string+"\n")



### Palette generator ###
@dataclass
class PalettePreset:#
	sample_method: int = 2

	reserve_transparent: bool = True 
	hex_pre_colors: List[[str,bool]] = None # ["#0123abc",...]
	img_pre_colors: str = None #file name to existing color palette
	img_fixed_mask: str = None #file name to fixed mask white=fixed black=movable color
	
	max_colors: int = 64		#Max allowed colors including transparency
	gray_count: int = None	#Grayscale color count, None = Auto
	hue_count:  int = 12		#Split Hues in this many buckets

	min_sat: float = 0.0 	#min/max ranges are percentages
	max_sat: float = 1.0
	min_lum: float = 0.0
	max_lum: float = 1.2

	packing_fac: float = 1.0 #Packing efficiency
	max_attempts: int = 1024 #After this many max_attempts per point, point Sampler will give up
	relax_count: int = 64 #number of relax iteration after point sampling
	
	seed: int = 0 # 0=random run to run


class PaletteGenerator:
	"""
		Generate palette where the colors are perceptually evenly spaced out in OKLab colorspace
	"""
	OKLAB_GAMUT_VOLUME = 0.054197416

	def __init__(self, preset: [PalettePreset]):
		self.point_grid = None
		self.point_radius = None

		if preset is None:
			preset = PalettePreset()
	
		self.p = PalettePreset(**{k: getattr(preset, k) for k in preset.__dataclass_fields__})

		if self.p.packing_fac <= 0:
			self.p.packing_fac = 1e-12

		if self.p.max_colors:
			self.p.max_colors -= self.p.reserve_transparent #max_count includes transparency
		else:
			self.p.max_colors = 64
	
		self.rand = RorrLCG(self.p.seed) if self.p.seed != None else RorrLCG()


	def getRandOklab(self):
		rand_p = KaelColor("OKLAB")
		while not rand_p.inOklabGamut():
			rand_p.col = [self.rand.f(), self.rand.f()-0.5, self.rand.f()-0.5]
	
		return rand_p

	def applyColorLimits(self):
		apply_luminosity = self.p.max_lum!=1.0 or self.p.min_lum!=0.0
		apply_saturation = self.p.max_sat!=1.0 or self.p.min_sat!=0.0
		count=0
		max_chroma = math.sqrt(0.5**2+0.5**2) 
		for p in self.point_grid.cloud:
			if apply_luminosity:
				lum_width = self.p.max_lum - self.p.min_lum
				p.col[0] = p.col[0]*lum_width + self.p.min_lum
		
			if apply_saturation and not p.isOkSrgbGray():
				sat_width = self.p.max_sat - self.p.min_sat
				chroma = p.calcChroma()
	
				rel_sat = chroma / max_chroma
				scaled_sat = (rel_sat * sat_width + self.p.min_sat) * max_chroma

				col_vec = [p.col[1], p.col[2]] #2D Vector a,b 
				col_vec = [col_vec[0]/chroma, col_vec[1]/chroma] #Normalize
				col_vec = [col_vec[0]*scaled_sat, col_vec[1]*scaled_sat] #Scale
				p.col = [p.col[0], col_vec[0], col_vec[1]]

	def addPreColors(self, alpha_threshold:int=int(0), fixed_mask_threshold:int=int(128)):
		#Add uint8 colors from image
		if self.p.img_pre_colors != None:
		
			pre_palette = Image.open(self.p.img_pre_colors)
			pre_palette = pre_palette.convert('RGBA')
			rgba_list = list(pre_palette.getdata())
	
			fixed_list = [0]*len(rgba_list)
			if self.p.img_fixed_mask !=None:
				fixed_mask = Image.open(self.p.img_fixed_mask)
				fixed_mask = fixed_mask.convert('L')
				fixed_list = list(fixed_mask.getdata())
	
			index=0
			for col in rgba_list:
				if col[3] <= alpha_threshold:
					continue
				r,g,b,a = col
				r = float(r)/255.0
				g = float(g)/255.0
				b = float(b)/255.0
				a = float(a)/255.0
				is_fixed = fixed_list[index] > fixed_mask_threshold
				new_col = KaelColor( "SRGB", [r,g,b], alpha=a, fixed=is_fixed )
				new_col.toOklab()
				self.point_grid.insert(new_col)
				index+=1
	
		#Add hex colors, format ["#1234abcd",...] or [["abc123",is_fixed]...]
		if self.p.hex_pre_colors != None and len(self.p.hex_pre_colors):
			for hexlet_info in self.p.hex_pre_colors:
				hexlet,fixed = [hexlet_info[0], False]
				if len(hexlet)>1:
					hexlet, fixed = hexlet_info
	
				srgba=hexToSrgba(hexlet)
				if srgba[3] < alpha_threshold*255.0:
					continue
	
				oklab = srgba[:3]
				new_col = KaelColor( "SRGB", oklab, alpha=srgba[3], fixed=fixed )
				new_col.toOklab()
				self.point_grid.insert(new_col)


	### Oklab point sampler methods within gamut ###

	def zeroSampler(self):
		attempts=0
		while self.point_grid.length < self.p.max_colors:
			new_col = KaelColor("OKLAB",[0.5,0.0,0.0])
			self.point_grid.insert(new_col)

	#Simple rejection sampling
	def randomSampler(self, min_dist):
		attempts=0
		while self.point_grid.length < self.p.max_colors and attempts < self.p.max_attempts:
			attempts+=1
			new_col = self.getRandOklab()
			neighbor = self.point_grid.findNearest(new_col, self.point_radius)
			if neighbor == None or neighbor.dist_sq>=min_dist**2:
				self.point_grid.insert(new_col)
		print("randomSampler Loop count "+str(attempts))

	#Generate grayscale gradient between black and white
	def generateGrays(self):
		if self.p.gray_count == None:
			self.p.gray_count = int(round(1.0/(self.point_radius)))
		if self.p.gray_count:
			darkest_black = KaelColor( "SRGB", [0.499/255,0.499/255,0.499/255] ).calcLum()
			self.p.gray_count = min(self.p.max_colors - len(self.point_grid.cloud), self.p.gray_count)
			#Use minimum starting luminosity that second darkest black isn't so close to 0 
			for i in range(0,self.p.gray_count):
				denom = max(1, self.p.gray_count-1)
				lum = float(i)/((denom))
				scale = (denom-i)/denom
				lum+= darkest_black*scale #Fade that brightest remains 1.0
				new_point = [lum,0,0]
				new_col = KaelColor( "OKLAB", new_point, 1.0, True )
				self.point_grid.insert(new_col)
	
	def populatePointCloud(self, histogram_path=None):	
		unit_volume = self.OKLAB_GAMUT_VOLUME/max(1,self.p.max_colors)
		cell_size = unit_volume**(1.0/3.0) 
		self.point_radius = cell_size * self.p.packing_fac
		print("Using point_radius "+str(round(self.point_radius,4)))

		oklabRange = [[0.0,-0.5,-0.5],[1.0,0.5,0.5]]
		self.point_grid = PointGrid(cell_size, oklabRange )

		self.addPreColors()
		self.generateGrays()

		if self.p.sample_method in [0,2]:
			self.randomSampler(self.point_radius*0.51)
		if self.p.sample_method in [1,2]:
			self.zeroSampler()

		simulator = ParticleSim(self.rand, self.point_grid)
		simulator.relaxCloud(
			iterations=self.p.relax_count,
			approx_radius = self.point_radius,
			record_frames = histogram_path,
   	)

		self.applyColorLimits()

		return self.point_grid.cloud



	### Palette processing ###
	def paletteToHex(self):
		hex_list = []
		if self.p.reserve_transparent:
				hex_list.append("#00000000")
	
		for p in self.point_grid.cloud:
			hex_list.append(p.getSrgbHex())
		
		return hex_list

	def paletteToImg(self, filename: str = "palette.png"):
		rgba = []
		if self.p.reserve_transparent:
			rgba.append((0, 0, 0, 0))
	
		for p in self.point_grid.cloud:
			r,g,b = p.asSrgb()
			if not valid_vec3([r,g,b]):
				continue
			r = min( max( int(round(r * 255.0)), 0 ), 255) 
			g = min( max( int(round(g * 255.0)), 0 ), 255) 
			b = min( max( int(round(b * 255.0)), 0 ), 255) 
			rgba.append((r, g, b, 255))
	
		if len(rgba) == 0:
			return None

		arr = np.array([rgba], dtype=np.uint8)
		img = Image.fromarray(arr, mode="RGBA")
		img.save(filename)
		return img


	def separateGrays(self, KaelColor_array = list [KaelColor]):
		gray_colors = []
		hued_colors = []

		for p in KaelColor_array:
			if p.isOkSrgbGray():
				gray_colors.append(p)
			else:
				hued_colors.append(p)
		
		return gray_colors, hued_colors

	def sortByLum(self, KaelColor_array: list[KaelColor]):
		return sorted(KaelColor_array, key=lambda x: x.col[0])

	def sortPalette(self):
		gray_colors, hued_colors = self.separateGrays(self.point_grid.cloud)
	
		#Place hues in same buckets
		hue_buckets = [[] for _ in range(self.p.hue_count)]
		hue_bucket_width = 2*math.pi * (1.0/self.p.hue_count)

		for p in hued_colors:
			col_hue = math.atan2(p.col[2], p.col[1]) + 2* math.pi
			bucket_index = int(col_hue/hue_bucket_width) % self.p.hue_count
			hue_buckets[bucket_index].append(p)

		#Sort hue buckets by luminance
		sorted_hue_buckets = []
		for bucket in hue_buckets:
			sorted_bucket = self.sortByLum(bucket)
			sorted_hue_buckets.append(sorted_bucket)

		#combine colors into single array
		sorted_colors = []

		sorted_grays = self.sortByLum(gray_colors)
		for p in sorted_grays:
			sorted_colors.append(p)

		for bucket in sorted_hue_buckets:
			for p in bucket:
				sorted_colors.append(p)
	
		self.point_grid.cloud = sorted_colors

	#EOF PaletteGenerator

