#PointList.py
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field

from palette.PalettePreset import *
from palette.OkTools import *


PointType = np.dtype([
	('color', float, 3),
	('alpha', float),
	('fixed', bool)
])


class PointList:

	TYPE_STR = ("srgb","linear","oklab")
	INVALID_COL = (1.0,0.0,0.5)

	type: str = None
	points: PointType = None
	preset: PalettePreset = None
	rand: np.random.Generator = None

	def __init__(self,
		color_space: str = "oklab",
		point_count: int = 0,
		alpha: float = 1.0,
		fixed: bool = False,
		preset: PalettePreset = None
	):
		self.type = str(color_space).lower()
		if self.type not in self.TYPE_STR:
			print("Invalid type" + self.type)
			self.type = None

		self.points = np.zeros((point_count) ,dtype=PointType)
		self.points["alpha"] = alpha
		self.points["fixed"] = fixed

		#Optional preset
		if preset != None:
			self.preset = preset
			self.rand = np.random.default_rng(self.preset.seed)


	def __len__(self):
		return len(self.points["color"])

	def copy(self):
		new = PointList(self.type)
		new.points = self.points.copy()
		new.preset = self.preset
		new.rand = self.rand
		return new

	### Return points["color"] as

	def getSrgbColor(self):
		if self.type == "linear":
			return linearToSrgb(self.points["color"])
		if self.type == "oklab":
			return oklabToSrgb(self.points["color"])
		if self.type == "srgb":
			return self.points["color"]
		return None

	def getLinearColor(self):
		if self.type == "linear":
			return self.points["color"]
		if self.type == "oklab":
			return oklabToLinear(self.points["color"])
		if self.type == "srgb":
			return srgbToLinear(self.points["color"])
		return None

	def getOklabColor(self):
		if self.type == "linear":
			return linearToOklab(self.points["color"])
		if self.type == "oklab":
			return self.points["color"]
		if self.type == "srgb":
			return srgbToOklab(self.points["color"])
		return None

	def getAsColor(self, type):
		if type == "srgb":
			return self.getSrgbColor()
		if type == "linear":
			return self.getLinearColor()
		if type == "oklab":
			return self.getOklabColor()
		return None


	#concatenate one or array of point_list.points to self, convert if necessary
	def concat(self, new_stack):
		if isinstance(new_stack, PointList):
			new_stack = [new_stack]

		same_type_list = [self.points]
		for current_list in new_stack:
			if self.type == current_list.type:
				safe_list = current_list
			else:
				safe_list = current_list.copy()
				safe_list.points["color"] = safe_list.getAsColor(self.type)

			if safe_list.points["color"] is None:
				print("Warning: PointList has no type! No concat is done.")
				continue

			same_type_list.append(safe_list.points)

		self.points = np.concatenate(same_type_list)


	#apply luminosity and saturation adjustments
	def applyColorLimits(self):
		if self.preset == None:
			print("Preset is unset")

		apply_luminosity = self.preset.max_lum!=1.0 or self.preset.min_lum!=0.0
		apply_saturation = self.preset.max_sat!=1.0 or self.preset.min_sat!=0.0

		max_chroma = np.sqrt(0.5**2+0.5**2)

		not_fixed = ~self.points["fixed"]
		color_list = self.points["color"][not_fixed]

		if apply_luminosity:
			lum_width = self.preset.max_lum - self.preset.min_lum
			color_list[:,0] = color_list[:,0]*lum_width + self.preset.min_lum
	
		if apply_saturation:
			hued_idxs = ~(OkTools.isOkSrgbGray(color_list))
			hued_colors = color_list[hued_idxs]

			sat_width = self.preset.max_sat - self.preset.min_sat
			chroma = OkTools.calcChroma(hued_colors)

			rel_sat = np.zeros_like(chroma)
			rel_sat = chroma/max_chroma
			scaled_sat = (rel_sat * sat_width + self.preset.min_sat) * max_chroma

			col_vec = hued_colors[:,1:3] #2D Vector a,b
			col_vec = col_vec/chroma[:,None] #Normalize
			col_vec = col_vec*scaled_sat[:,None] #Scale
			color_list[hued_idxs,1:3] = col_vec

		self.points["color"][not_fixed] = color_list


	#Sort into hue buckets
	def sortPalette(self, hue_count: int = None):
		if hue_count == None and self.preset != None:
			hue_count = self.preset.hue_count #default, arg overrides
		hue_count = max(1, hue_count)

		color_list = self.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list)
	
		#bucket similar hues, sort each bucket by luminosity
		hue_bucket_width = 2*np.pi * (1.0/hue_count)

		color_list_hue = np.atan2(color_list[:,2], color_list[:,1]) + 2* np.pi
		hue_bucket_idxs = color_list_hue/hue_bucket_width
		hue_bucket_idxs = hue_bucket_idxs.astype(int) % hue_count #hue_bucket_idxs[i] = hue_idx of color_list[i]
		hue_bucket_idxs[is_gray] = -1 #grayscale first

		sorted_idxs = np.array([],dtype=int)
		for idx in range(-1, hue_count):
			this_bucket_idxs = np.where(hue_bucket_idxs==idx)[0] #bucket colors with same hue_idx
			bucket_colors = color_list[this_bucket_idxs]
			sorted_sub_idxs= np.argsort(bucket_colors[:, 0]) #sort bucket by luminosity
			this_bucket_sorted_idxs = this_bucket_idxs[sorted_sub_idxs]

			sorted_idxs = np.concatenate([sorted_idxs, this_bucket_sorted_idxs])

		self.points = self.points[sorted_idxs]


	#Save to file
	def saveAsImage(self):
		if self.preset == None:
			print("Warning: Preset is unset")
			return None

		p_count = len(self)
		if(p_count==0 and self.preset.reserve_transparent == 0):
			return

		rgba = np.zeros((p_count,4))

		rgba[:,:3] = self.getSrgbColor()
		rgba[:,3] = self.points["alpha"]

		if self.preset.reserve_transparent:
			rgba = np.insert(rgba, 0, np.array([0, 0, 0, 0]),axis=0)

		rgba = np.round(rgba*255.0)
		rgba = np.clip(rgba, 0, 255.0)

		arr = np.array([rgba], dtype=np.uint8)
		img = Image.fromarray(arr, mode="RGBA")

		img.save(self.preset.palette_output)
		return img