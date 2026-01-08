#PointList.py
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field


PointType = np.dtype([
	('color', float, 3), 
	('alpha', float), 
	('fixed', bool)
])


@dataclass
class PointList:
	"""void PointList(string color_space, int point_count = 0)"""

	TYPE_STR = ("srgb","linear","oklab")
	INVALID_COL = (1.0,0.0,0.5)

	type: str
	points: PointType #For clarity we call these points as they contain more info than just the color

	def __init__(self, color_space: str, point_count: int = 0, alpha: float = 1.0, fixed: bool = False):
		self.type = str(color_space).lower()
		if self.type not in self.TYPE_STR:
			print("Invalid type" + self.type)
			self.type = None

		self.points = np.zeros((point_count),dtype=PointType)
		self.points["alpha"] = alpha
		self.points["fixed"] = fixed

	#public

	def __len__(self):
		return len(self.points["color"])

	def concat(self, new_list):
		"""void concat(PointList new_list, bool silent=false)"""
		if self.type != new_list.type:
			raise TypeError("Concatenating different types") 
		self.points = np.concatenate([self.points, new_list.points])