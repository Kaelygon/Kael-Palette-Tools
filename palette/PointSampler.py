#PointSampler.py
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

from palette.PointList import *
from palette.OkTools import *
from palette.PalettePreset import *

#Generate PointList.points within OKLab gamut and colorspace
class PointSampler:
	#query np_points vs other_points, remove points np_points that are closer than min_dist to other_points
	@staticmethod
	def _removeNearbyPoints(np_points, other_points, min_dist):
		if len(np_points) == 0 or len(other_points) == 0:
			return np_points
		point_tree = cKDTree(other_points)
		dists, _ = point_tree.query(np_points, k=1)
		not_near = dists >= min_dist
		np_points = np_points[not_near]
		return np_points


	#query np_points vs np_points, remove other point that's closer than min_dist
	@staticmethod
	def _removeNearbySelf(np_points, min_dist):
		n = len(np_points)
		if n < 2:
			return np_points

		tree = cKDTree(np_points)
		not_near = np.zeros(n, dtype=bool) + True
		idxs = tree.query_ball_point(np_points, min_dist)

		for i in range(n):
			if not not_near[i]:
					continue
			for j in idxs[i]:
					if j != i:
						not_near[j] = False

		np_points = np_points[not_near]
		return np_points


	@staticmethod
	def _calcBatchSize(point_count, radius, overlap):
		box_volume = np.prod(OkTools.OKLAB_BOX_SIZE)
		in_gamut_prob = OkTools.OKLAB_GAMUT_VOLUME / box_volume #probability being in gamut

		min_dist = None
		batch_mul = 1.0
		if overlap!=None and radius != None:
			overlap = min(max(overlap, 0.05), 0.95)
			batch_mul = 1.0 / overlap #guesstimate point_count + reject_count of random point_cloud
			min_dist = radius - radius * overlap

		batch_size = int(batch_mul * point_count / in_gamut_prob) +1
		return batch_size, min_dist


	@staticmethod
	def _createMeshGrid(min_dist):
		eps = 0.05
		#Create mesh grid masked by oklab gamut
		l_min, a_min, b_min = OkTools.OKLAB_BOX_MIN +eps
		l_max, a_max, b_max = OkTools.OKLAB_BOX_MAX -eps

		lab_size = OkTools.OKLAB_BOX_SIZE - 2.0*eps
		l_steps, a_steps, b_steps, = np.round(lab_size/min_dist).astype(int)

		#mesh grid
		l = np.linspace(l_min, l_max, num=l_steps, endpoint=True)
		a = np.linspace(a_min, a_max, num=a_steps, endpoint=True)
		b = np.linspace(b_min, b_max, num=b_steps, endpoint=True)

		ll,aa,bb = np.meshgrid(l, a, b, indexing='ij')
		grid_points = np.vstack([ll.ravel(), aa.ravel(), bb.ravel()]).T

		in_gamut = OkTools.inOklabGamut(grid_points)
		grid_points = grid_points[in_gamut]
		grid_points, _ = OkTools.clipToOklabGamut(grid_points)

		return grid_points



	#convert preset.img_pre_colors to PointList. Luminance in preset.img_fixed_mask makes the point immovable
	@staticmethod
	def precolor(preset: PalettePreset, alpha_threshold:int=0, fixed_mask_threshold:int=128):
		if preset.img_pre_colors == None:
			print("Warning: preset.img_pre_colors is not set")
			return PointList("oklab", 0)

		#Add uint8 colors from image
		pre_palette = Image.open(preset.img_pre_colors)
		pre_palette = pre_palette.convert('RGBA')
		rgba_list = np.array(pre_palette.getdata())
		rgba_list = rgba_list.astype(float) / np.iinfo(np.uint8).max

		lab_list = srgbToOklab(rgba_list[:,:3])
		alpha_list = rgba_list[:,3]

		fixed_list = np.zeros(len(lab_list), dtype=bool) + True #if no fixed_mask, default=fixed
		if preset.img_fixed_mask !=None:
			fixed_mask = Image.open(preset.img_fixed_mask)
			fixed_mask = fixed_mask.convert('L')
			fixed_mask = np.array(fixed_mask).flatten()
			valid_count = min(len(fixed_list),len(fixed_mask))
			fixed_list[:valid_count] = fixed_mask[:valid_count] > fixed_mask_threshold

		if alpha_threshold!=None:
			#keep only opaque
			opaque = alpha_list > alpha_threshold
			lab_list = lab_list[opaque]
			fixed_list = fixed_list[opaque]

		pre_list = PointList("oklab", len(lab_list), alpha=1.0)
		pre_list.points["color"] = lab_list
		pre_list.points["fixed"] = fixed_list

		return pre_list

	def _generateEdges(num):
		steps = np.arange(num)
		edges = []

		for a in (0, num - 1):
			for b in (0, num - 1):
					edges.append(np.stack([steps, np.full(num, a), np.full(num, b)], axis=1))
					edges.append(np.stack([np.full(num, a), steps, np.full(num, b)], axis=1))
					edges.append(np.stack([np.full(num, a), np.full(num, b), steps], axis=1))

		edges = np.vstack(edges)
		edges = edges / max(1,np.max(edges))
		ok_corners = linearToOklab(edges)
		return ok_corners



	### Poisson disk sampling ###
	@staticmethod
	def poissonDisk(
		point_list: PointList,
		point_count: int,
		radius: float,
		overlap: float = 0.0,
		generate_edges: bool = False,
		kissing_number: int = 24,
		sample_attempts: int = 1000,
		logging = False
	):
		initial_points = point_list.points["color"]
		min_dist = radius - radius * overlap

		offset = np.zeros((1,3))
		rand = point_list.rand
		if rand != None:
			offset = rand.random((1,3))
		neighbor_offsets = OkTools.sphereNormals(kissing_number, offset) * radius

		accepted_points = np.empty((0,3))

		if len(initial_points) == 0 or generate_edges:
			ok_corners = PointSampler._generateEdges(256)

			#remove edges close to initial_points
			ok_corners = PointSampler._removeNearbyPoints(ok_corners, initial_points, radius)
			ok_corners = PointSampler._removeNearbySelf(ok_corners, radius)
			accepted_points = np.vstack((accepted_points, ok_corners))

		active_points = np.vstack((accepted_points, initial_points))


		iterations = 0
		while accepted_points.shape[0] < point_count and active_points.shape[0]:
			if iterations >= sample_attempts:
				break
			iterations+=1
			#Generate sphere of points around each point
			np_points = active_points[:, None, :] + neighbor_offsets[None, :, :]
			np_points = np_points.reshape(-1, 3)

			#remove outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			#remove nearby points
			all_points = np.vstack((accepted_points, initial_points))
			np_points = PointSampler._removeNearbyPoints(np_points, all_points, min_dist)
			np_points = PointSampler._removeNearbySelf(np_points, min_dist)

			#new points become new seeds
			accepted_points = np.vstack((accepted_points, np_points))
			active_points = np_points

		#to PointList
		accepted_points = accepted_points[:point_count]
		if len(accepted_points):
			output_list = PointList("oklab", len(accepted_points), 1.0, False)
			output_list.points['color'] = accepted_points
		else:
			output_list = PointList("oklab", 0)

		if logging:
			print("PointSampler poissonDisk() finished after " + str(iterations) + " iterations. " +
				str(len(accepted_points)) + " points.")
		return output_list



	#### Random Reject by distance ###
	@staticmethod
	def randomReject(
		point_list: PointList,
		point_count: int,
		radius: float = None,
		overlap: float = None, #None skips distance check
		sample_attempts: int = 1000,
		logging = False
	):
		rand = point_list.rand
		if rand == None:
			rand = np.random.default_rng(0)

		batch_size, min_dist = PointSampler._calcBatchSize(point_count, radius, overlap)

		initial_points = point_list.points["color"]
		accepted_points = np.empty((0,3))

		attempts=0
		while len(accepted_points) < point_count:
			if attempts >= sample_attempts:
				break
			attempts+=1

			np_points = rand.random((batch_size, 3))*OkTools.OKLAB_BOX_SIZE + OkTools.OKLAB_BOX_MIN

			#discard outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			if len(np_points) == 0:
				continue

			#discard if too close
			if min_dist != None:
				all_points = np.vstack((accepted_points, initial_points))
				np_points = PointSampler._removeNearbyPoints(np_points, all_points, min_dist)
				np_points = PointSampler._removeNearbySelf(np_points, min_dist)

			accepted_points = np.vstack((accepted_points, np_points))

		#to PointList
		accepted_points = accepted_points[:point_count]
		if len(accepted_points):
			output_list = PointList("oklab", len(accepted_points), 1.0, False)
			output_list.points['color'] = accepted_points
		else:
			output_list = PointList("oklab", 0)

		if logging:
			print("PointSampler randomReject() finished after " + str(attempts) + " attempts. " +
				str(len(accepted_points)) + " points.")
		return output_list
		


	#### Grid with jitter ###
	@staticmethod
	def gridReject(
		point_list: PointList,
		point_count: int,
		radius: float = None,
		overlap: float = None,
		kissing_number = 48,
		sample_attempts: int = 1000,
		logging = False
	):
		rand = point_list.rand
		if rand == None:
			rand = np.random.default_rng(0)

		min_dist = radius - radius * overlap
		grid_points = PointSampler._createMeshGrid(radius/2.0) #twice as dense as search grid

		initial_points = point_list.points["color"]
		accepted_points = np.empty((0,3))

		offset_list = []
		steps = max(1,int(sample_attempts/kissing_number))
		radius_steps = np.linspace(1.0/steps, (steps-1)/steps, num=steps,endpoint=True) * radius
		for j in range(steps):
			neighbor_offsets = OkTools.sphereNormals(kissing_number, radius_steps[j]) * radius_steps[j]
			offset_list.append(neighbor_offsets)
		neighbor_offsets = np.vstack(offset_list)

		#Accept order matters so this acts as jitter
		order = np.arange(neighbor_offsets.shape[0])
		rand.shuffle(order)
		neighbor_offsets = neighbor_offsets[order]

		attempts=0
		for offset in neighbor_offsets:
			if attempts >= sample_attempts:
				break
			attempts+=1
			if accepted_points.shape[0] >= point_count:
				break

			np_points = grid_points + offset

			#discard outside gamut
			in_gamut = OkTools.inOklabGamut(np_points)
			np_points = np_points[in_gamut]

			if len(np_points) == 0:
				continue

			#discard if too close
			all_points = np.vstack((accepted_points, initial_points))
			np_points = PointSampler._removeNearbyPoints(np_points, all_points, min_dist)
			np_points = PointSampler._removeNearbySelf(np_points, min_dist)

			accepted_points = np.vstack((accepted_points, np_points))


		#to PointList
		accepted_points = accepted_points[:point_count]
		if len(accepted_points):
			output_list = PointList("oklab", len(accepted_points), 1.0, False)
			output_list.points['color'] = accepted_points
		else:
			output_list = PointList("oklab", 0)

		if logging:
			print("PointSampler gridReject() finished after " + str(attempts) + " attempts. "
				+ str(len(accepted_points)) + " points.")
		return output_list
		

	#### Add points to origin ###
	@staticmethod
	def zero(
		point_count: int,
		origin: list[float] = [0.5, 0.0, 0.0],
		logging = False
	):
		accepted_points = PointList("oklab", point_count, 1.0, False)
		accepted_points.points['color'] = origin

		if logging:
			print("PointSampler zero() generated " + str(len(accepted_points)) + " points.")

		return accepted_points


	### Generate grayscale gradient between black and white ###
	@staticmethod
	def grayscale(
		point_count: int = None,
		point_radius: float = None,
		minimum: float=0.0,
		maximum: float=1.0
	):
		if point_count == None and point_radius:
			point_count = int(round(1.0/(point_radius)))

		lab = np.zeros((point_count,3))
		lab[:,0] = np.linspace(minimum,maximum,num=point_count,endpoint=True)

		accepted_points = PointList("oklab", point_count, alpha=1.0, fixed=True )
		accepted_points.points['color'] = lab

		return accepted_points
