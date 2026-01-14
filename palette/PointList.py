import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .OkLab import OkLab
from .OkTools import OkTools



class PointList_Stats:
	@staticmethod
	def _printPairStats(color_list, print_count, listName = "", precision = 4):
		"""void _printPairStats(float[][3] color_list, int print_count, string listName = "", int precision = 4)"""
		if len(color_list)<2:
			return

		color_tree = cKDTree(color_list)
		dists, idxs = color_tree.query(color_list, k=2)

		pair_idxs = idxs[:,:2]
		pair_dists = dists[:,1]

		#remove redundant
		pair_idxs = np.sort(pair_idxs, axis=1) #[a,b] [b,a] -> [a,b] [a,b]
		_, unique_idxs = np.unique(pair_idxs, axis=0, return_index=True)

		pair_idxs = pair_idxs[unique_idxs]
		pair_dists = pair_dists[unique_idxs]

		#sort by distance
		sorted_idxs = np.argsort(pair_dists)
		pair_idxs = pair_idxs[sorted_idxs]
		pair_dists = pair_dists[sorted_idxs]

		#srgb is only used for hex printing
		srgb_list = OkLab.oklabToSrgb(color_list)
		srgb_pairs = srgb_list[pair_idxs]

		#Print smallest and biggest pair gaps and their hex values
		print(listName+" Closest pairs")
		for i, pair_idx in enumerate(pair_idxs[:print_count]):
			srgb_pair = srgb_pairs[i]
			str_first=OkTools.srgbToHex(srgb_pair[0])
			str_second=OkTools.srgbToHex(srgb_pair[1])

			pair_dist = pair_dists[i]
			print("d:" + str(round(pair_dist,precision))+" "+str_first+" "+str_second)
		print("")

		#print only unseen values
		far_start = max(print_count, len(pair_idxs)-print_count)
		if far_start < len(pair_idxs):
			print(listName+" Farthest pairs")
			for i, pair in enumerate(pair_idxs[far_start:], far_start):
				srgb_pair = srgb_pairs[i]
				str_first=OkTools.srgbToHex(srgb_pair[0])
				str_second=OkTools.srgbToHex(srgb_pair[1])

				pair_dist = pair_dists[i]
				print("d:" + str(round(pair_dist,precision))+" "+str_first+" "+str_second)


		#Average and median
		avg_gap = np.average(pair_dists)
		median_gap = np.median(pair_dists)

		print(listName+" Avg pair gap: "		+str(round( avg_gap, precision )) )
		print(listName+" Median pair gap: "	+str(round( median_gap, precision )) )
		print("")


		#Relative gap delta
		smallest_gap = pair_dists[0]
		largest_gap = pair_dists[-1]

		gap_delta = abs(1.0 - smallest_gap/avg_gap)
		print(listName+" smallest gap delta to avg "+str(round(100*gap_delta,precision))+" %")
		gap_delta= abs(1.0 - largest_gap/avg_gap)
		print(listName+" biggest gap delta to avg  "+str(round(100*gap_delta,precision))+" %")
		print("")


	@staticmethod
	def gaps(point_list, print_count = 4, listName = "", precision = 4):
		"""void printGapStats(PointList point_list, int print_count = 4, string listName = "", int precision = 4)"""
		color_list = point_list.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list) 
		gray_colors = color_list[is_gray] 
		hued_colors = color_list[~is_gray]

		PointList.Stats._printPairStats(gray_colors,print_count,"Grayscale")
		PointList.Stats._printPairStats(hued_colors,print_count,"Chroma")





class PointList_Color:

	### color conversion ###

	@staticmethod
	def getSrgb(point_list):
		if point_list.type == "linear":
			return OkLab.linearToSrgb(point_list.points["color"])
		if point_list.type == "oklab":
			return OkLab.oklabToSrgb(point_list.points["color"])
		if point_list.type == "srgb":
			return point_list.points["color"]
		return None

	@staticmethod
	def getLinear(point_list):
		if point_list.type == "linear":
			return point_list.points["color"]
		if point_list.type == "oklab":
			return OkLab.oklabToLinear(point_list.points["color"])
		if point_list.type == "srgb":
			return OkLab.srgbToLinear(point_list.points["color"])
		return None

	@staticmethod
	def getOklab(point_list):
		if point_list.type == "linear":
			return OkLab.linearToOklab(point_list.points["color"])
		if point_list.type == "oklab":
			return point_list.points["color"]
		if point_list.type == "srgb":
			return OkLab.srgbToOklab(point_list.points["color"])
		return None

	@staticmethod
	def getAs(point_list, type):
		"""Return PointList.points["color"] as linear, oklab or srgb"""
		if type == "srgb":
			return PointList.Color.getSrgb(point_list)
		if type == "linear":
			return PointList.Color.getLinear(point_list)
		if type == "oklab":
			return PointList.Color.getOklab(point_list)
		return None



	### color filtering ###

	#apply luminosity and saturation adjustments to PointList.points["color"]
	@staticmethod
	def applyLimits(point_list, preset):
		if preset == None:
			print("Preset is unset")

		apply_luminosity = preset.max_lum!=1.0 or preset.min_lum!=0.0
		apply_saturation = preset.max_sat!=1.0 or preset.min_sat!=0.0

		max_chroma = np.sqrt(0.5**2+0.5**2)

		not_fixed = ~point_list.points["fixed"]
		color_list = point_list.points["color"][not_fixed]

		if apply_luminosity:
			lum_width = preset.max_lum - preset.min_lum
			color_list[:,0] = color_list[:,0]*lum_width + preset.min_lum
	
		if apply_saturation:
			hued_idxs = ~(OkTools.isOkSrgbGray(color_list))
			hued_colors = color_list[hued_idxs]

			sat_width = preset.max_sat - preset.min_sat
			chroma = OkTools.calcChroma(hued_colors)

			rel_sat = np.zeros_like(chroma)
			rel_sat = chroma/max_chroma
			scaled_sat = (rel_sat * sat_width + preset.min_sat) * max_chroma

			col_vec = hued_colors[:,1:3] #2D Vector a,b
			col_vec = col_vec/chroma[:,None] #Normalize
			col_vec = col_vec*scaled_sat[:,None] #Scale
			color_list[hued_idxs,1:3] = col_vec

		new_list = point_list.copy()
		new_list.points["color"][not_fixed] = color_list
		return new_list

	#Sort into hue buckets
	@staticmethod
	def sort(point_list, preset):
		hue_count = preset.hue_count
		hue_count = max(1, hue_count)

		color_list = point_list.points["color"]
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

		new_list = point_list.copy()
		new_list.points = point_list.points[sorted_idxs]
		return new_list



PointList_PointType = np.dtype([
	('color', float, 3),
	('alpha', float),
	('fixed', bool)
])

class PointList:
	#Namespacing
	Color = PointList_Color
	PointType = PointList_PointType
	Stats = PointList_Stats

	#Consts
	TYPE_STR = ("srgb","linear","oklab")
	INVALID_COL = (1.0,0.0,0.5)

	#class vars
	type: str = None
	points: PointType = None

	def __init__(self,
		color_space: str = "oklab",
		point_count: int = 0,
		alpha: float = 1.0,
		fixed: bool = False,
	):
		self.type = str(color_space).lower()
		if self.type not in self.TYPE_STR:
			print("Invalid type" + self.type)
			self.type = None

		self.points = np.zeros((point_count) ,dtype=PointList.PointType)
		self.points["color"] = self.INVALID_COL
		self.points["alpha"] = alpha
		self.points["fixed"] = fixed


	def __len__(self):
		return len(self.points["color"])

	def copy(self):
		new = PointList(self.type)
		new.points = self.points.copy()
		return new


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
				safe_list.points["color"] = PointList.Color.getAs(current_list, self.type)

			if safe_list.points["color"] is None:
				print("PointList.concat: Type is not set! No concat is done.")
				continue

			same_type_list.append(safe_list.points)

		self.points = np.concatenate(same_type_list)


	#Save to file
	def saveImage(self, preset):
		if preset == None:
			print("PointList.saveImage: Preset is unset")
			return None

		p_count = len(self)
		if(p_count==0 and preset.reserve_transparent == 0):
			return

		rgba = np.zeros((p_count,4))

		rgba[:,:3] = PointList.Color.getAs(self, "srgb")
		rgba[:,3] = self.points["alpha"]

		if preset.reserve_transparent:
			rgba = np.insert(rgba, 0, np.array([0, 0, 0, 0]),axis=0)

		rgba = np.round(rgba*255.0)
		rgba = np.clip(rgba, 0, 255.0)

		arr = np.array([rgba], dtype=np.uint8)
		img = Image.fromarray(arr, mode="RGBA")

		img.save(preset.palette_output)
		return img