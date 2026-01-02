#PointList.py
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional


PointType = [
	('color', float, 3), 
	('alpha', float), 
	('fixed', bool)
]

#SoA point cloud to replace pointGrid+KaelColor
@dataclass
class PointList:
	TYPE_STR = ("srgb","linear","oklab")
	INVALID_COL = (1.0,0.0,0.5)

	type: str
	points: PointType #For clarity we call these points as they contain more info than just the color

	def __init__(self, color_space: str, point_count: int = 0):
		self.type = str(color_space).lower()
		if self.type not in self.TYPE_STR:
			print("Invalid type" + self.type)
			self.type = None

		self.points = np.zeros((point_count),dtype=PointType)

	#public

	#add new element at end of self.points
	def push(self, color, alpha=1.0, fixed=True):
		new_point = np.array([(color, alpha, fixed)], dtype=PointType)
		self.points = np.concatenate([self.points, new_point])

	def remove(self, idx):
		self.points = np.delete(self.points, idx, axis=0)

	def length(self):
		return len(self.points)

	def concat(self, new_list):
		self.points = np.concatenate([self.points, new_list.points])