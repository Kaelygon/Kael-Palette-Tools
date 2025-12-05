import math
from dataclasses import dataclass, field

from KaelColor import *
from float3 import *
from PointGrid import *



### palette analysis ###
class PointGridStats:
	#p0 has type (float3)[L,a,b]
	#pair_list has type (float,[],[])[dist, p0, p1]
	@staticmethod
	def _printPairStats(pair_list, print_count, listName="", calc_error=False):
		
		if len(pair_list) < 2:
			print("Not enough pairs for stats!")
			return
		
		precision = 4

		print(listName+" Closest pairs")
		for pair in pair_list[:print_count]:
			print(str(round(pair[0],precision))+" "+pair[1].getSrgbHex()+" "+pair[2].getSrgbHex())

		#print only unseen values
		far_start = max(print_count, len(pair_list)-print_count) 
		if far_start < len(pair_list):
			print(listName+" Farthest pairs")
		for pair in pair_list[far_start:]:
			print(str(round(pair[0],precision))+" "+pair[1].getSrgbHex()+" "+pair[2].getSrgbHex())

		#Average
		sumDist = 0.0
		for pair in pair_list:
			sumDist+=pair[0]
		avgDist = max(sumDist / len(pair_list),1e-12)

		#Median
		medianDist=0.0
		medianIndex=len(pair_list)//2
		if len(pair_list)%2==0:
			a = medianIndex
			b = max(medianIndex-1,0)
			medianDist = (pair_list[a][0] + pair_list[b][0]) / 2.0
		else:
			medianDist = pair_list[medianIndex][0]

		print(listName+" Avg pair distance: "+str(round(avgDist,precision)))
		print(listName+" Median pair distance: "+str(round(medianDist,precision)))
		print("")
		
		#Compare biggest gap to avg gap
		if calc_error == True:

			hued_pairs = [
				p for p in pair_list
				if not p[1].isOkSrgbGray() and not p[2].isOkSrgbGray()
			]
			hue_pair_count = len(hued_pairs)
			
			#Average hued only
			sumDist = 0.0
			for pair in hued_pairs:
				sumDist+=pair[0]
			avgHueDist = sumDist / hue_pair_count if hue_pair_count != 0 else 0.0001
		

			#All colors gaps
			all_smallest_gap = pair_list[0][0]
			all_largest_gap = pair_list[-1][0]
		

			#Hued colors gaps
			hued_smallest_gap = hued_pairs[0][0] if hued_pairs else None
			hued_largest_gap = hued_pairs[-1][0] if hued_pairs else None

			allError = abs(1.0 - all_largest_gap/avgDist)
			print("Biggest_gap  to avg_gap delta "+str(round(100*allError,precision))+" %")
			allError = abs(1.0 - all_smallest_gap/avgDist)
			print("Smallest_gap to avg_gap delta "+str(round(100*allError,precision))+" %")

			if hued_largest_gap:
				huedError = abs(1.0 - hued_largest_gap/avgHueDist)
				print("Hued biggest_gap  to avg_gap delta "+str(round(100*huedError,precision))+" %")
			if hued_largest_gap:
				huedError = abs(1.0 - hued_smallest_gap/avgHueDist)
				print("Hued smallest_gap to avg_gap delta "+str(round(100*huedError,precision))+" %")

		return


	# Input [[L,a,b], ...]
	# Find closest point for every point
	# Returns list of point pairs [[dist, p0, p1], ...] sorted by distance
	@staticmethod
	def _findClosestPairs(point_grid):
		if len(point_grid.cloud) < 2:
			return []
		dist_pair_array = []

		for p in point_grid.cloud:
			neighbor = point_grid.findNearest(p, point_grid.cell_size)
			if neighbor is None:
					continue
			if p == neighbor.point:
					continue
			dist_pair_array.append([math.sqrt(neighbor.dist_sq), p, neighbor.point])

		dist_pair_array.sort(key=lambda x: x[0])
		return dist_pair_array

	@staticmethod
	def printGapStats(point_grid, print_count):
		full_pair_arr = PointGridStats._findClosestPairs(point_grid)
		PointGridStats._printPairStats(full_pair_arr,print_count,"Full",1)
		print("")
