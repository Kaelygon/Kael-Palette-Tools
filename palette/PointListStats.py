#PointListStats.py
import math
import numpy as np
from scipy.spatial import cKDTree

from palette.PointList import *
from palette.OkTools import *

class PointListStats:
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

		# sort by distance
		sorted_idxs = np.argsort(pair_dists)
		pair_idxs = pair_idxs[sorted_idxs]
		pair_dists = pair_dists[sorted_idxs]

		#srgb is only used for hex printing
		srgb_list = oklabToSrgb(color_list)
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
	def printGapStats(point_list, print_count = 4, listName = "", precision = 4):
		"""void printGapStats(PointList point_list, int print_count = 4, string listName = "", int precision = 4)"""
		color_list = point_list.points["color"]
		is_gray = OkTools.isOkSrgbGray(color_list) 
		gray_colors = color_list[is_gray] 
		hued_colors = color_list[~is_gray]

		PointListStats._printPairStats(gray_colors,print_count,"Grayscale")
		PointListStats._printPairStats(hued_colors,print_count,"Chroma")
