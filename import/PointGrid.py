

import math
from dataclasses import dataclass, field

from KaelColor import *
from float3 import *


@dataclass
class NeighborData:
	point: KaelColor
	dist_vec: (float,float,float)
	dist_sq:	float

@dataclass
class NeighborList:
	root: KaelColor
	array: List[NeighborData]

class PointGrid:
	"""
		Store point cloud as a search PointGrid.grid and a 1D PointGrid.cloud
	"""
	INVALID_COLOR = float3(0.5,0.0,0.0)

	def __init__(self, point_radius: float, corners: [float3,float3] = None ):
		self.cloud: List[KaelColor] = []
		self.grid = {}
		self.length: int = 0
		self.cell_size: float = point_radius
		self.corners = corners
		if corners == None:
			self.max_cells = [[-1.0/self.cell_size]*3,[1.0/self.cell_size]*3]
		else:
			self.max_cells = [div_vec3(corners[0], [self.cell_size]*3),div_vec3(corners[1], [self.cell_size]*3)]
			self.max_cells = [roundAwayFromZero_vec3(self.max_cells[0]),roundAwayFromZero_vec3(self.max_cells[1])]

	def key(self, p: KaelColor):
		g_f = div_vec3(p.col,[self.cell_size]*3)
		if valid_vec3(g_f):
			return ( int(g_f[0]),int(g_f[1]),int(g_f[2]) )
		return [None,None,None]

	def insert(self, p: KaelColor):
		self.length+=1
		k = self.key(p)
		if k not in self.grid:
			self.grid[k] = []
		self.grid[k].append(p)
		self.cloud.append(p)

	def remove(self, p: KaelColor):
		k = self.key(p)

		if k in self.grid and p in self.grid[k]:
			self.length -= 1
			self.grid[k].remove(p)
			if not self.grid[k]:
				del self.grid[k]

		if p in self.cloud:
			self.cloud.remove(p)

	def setCol(self, p: KaelColor, q_col: float3):
		old_key = self.key(p)
		if old_key in self.grid and p in self.grid[old_key]:
			self.grid[old_key].remove(p)
			if not self.grid[old_key]:
					del self.grid[old_key]

		p.col = q_col

		new_key = self.key(p)
		if new_key not in self.grid:
			self.grid[new_key] = []
		self.grid[new_key].append(p)
		return p #return ref

	def cloudPosSnapshot(self):
		pts=[]
		for p in self.cloud:
			pts.append(list(p.col))
		return pts

	def findNeighbors(self, p: KaelColor, radius: float, neighbor_margin: float = 0.0):
		neighborhood = NeighborList(root=p, array=[])
		px, py, pz = p.col
		gx, gy, gz = self.key(p)
  
		cell_span = int(math.ceil((radius + neighbor_margin) / self.cell_size))
		x_r = [int(max(gx-cell_span, self.max_cells[0][0])), int(min(gx+cell_span, self.max_cells[1][0]) + 1 )]
		y_r = [int(max(gy-cell_span, self.max_cells[0][1])), int(min(gy+cell_span, self.max_cells[1][1]) + 1 )]
		z_r = [int(max(gz-cell_span, self.max_cells[0][2])), int(min(gz+cell_span, self.max_cells[1][2]) + 1 )]

		seen = set()
		seen.add(p)
		radius_sq = (radius + neighbor_margin)**2

		for dx in range(x_r[0], x_r[1]):
			for dy in range(y_r[0], y_r[1]):
				for dz in range(z_r[0], z_r[1]):
					nk = (dx, dy, dz)
					for q in self.grid.get(nk, ()):
							if q in seen:
								continue
							seen.add(q)
		
							dist_vec = (px-q.col[0], py-q.col[1], pz-q.col[2])
							length_sq = lengthSq_vec3((dist_vec[0],dist_vec[1],dist_vec[2]))
							if length_sq <= radius_sq:
								neighbor = NeighborData(point=q, dist_vec=dist_vec, dist_sq=length_sq)
								neighborhood.array.append(neighbor)
		return neighborhood

	def findNearest(self, p: KaelColor, point_radius: float, neighbor_margin: float = 0):
		neighbor = None
		closest=float('inf') 

		kx, ky, kz = self.key(p)
		for dx in (-1, 0, 1):
			for dy in (-1, 0, 1):
				for dz in (-1, 0, 1):
					nk = (kx + dx, ky + dy, kz + dz)
					chunk = self.grid.get(nk, ())
					for q in chunk:
						if q is p:
								continue
						dist_vec = sub_vec3(p.col, q.col)
						length_sq = lengthSq_vec3((dist_vec[0],dist_vec[1],dist_vec[2]))
						if length_sq < closest:
							closest = length_sq
							neighbor = NeighborData(point=q, dist_vec=dist_vec, dist_sq=length_sq)
		
		return neighbor

