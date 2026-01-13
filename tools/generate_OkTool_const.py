#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from dataclasses import dataclass, field

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palette import OkTools, OkLab

def createMeshGrid(size):
	x = np.arange(size)
	y = np.arange(size)
	z = np.arange(size)

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

	point_cloud = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
	return point_cloud / (size-1)

def calc_fallbackNorm(precision):
	sqrtThird = round((1.0/3.0)**0.5, precision)
	print("FALLBACK_NORM = np.array([", sqrtThird,"]*3) * [1,-1,1] #sqrt(1/3)" )

def calc_closestPoints(point_cloud, precision):
	ok_cloud = OkLab.srgbToOklab(point_cloud)

	tree = cKDTree(ok_cloud)
	dists, _ = tree.query(ok_cloud, k=2)
	OKLAB_8BIT_MARGIN = dists[:, 1].min()

	exp = 10**precision
	OKLAB_8BIT_MARGIN = np.floor(OKLAB_8BIT_MARGIN * exp) / exp

	print("OKLAB_8BIT_MARGIN = ", OKLAB_8BIT_MARGIN, " # minimum SRGB distance in oklab space" )


def calc_OkLabVolume(point_cloud, precision):
	in_gamut = point_cloud - [0.0, 0.5, 0.5]
	in_gamut = OkTools.inOklabGamut(in_gamut)

	OKLAB_GAMUT_VOLUME = np.sum(in_gamut) / point_cloud.shape[0]
	exp = 10**precision
	OKLAB_GAMUT_VOLUME = np.ceil(OKLAB_GAMUT_VOLUME * exp) / exp
	print("OKLAB_GAMUT_VOLUME = ", "%0.20f" % OKLAB_GAMUT_VOLUME, " # (oklab gamut) / (srgb gamut)" )


def calc_OkLabRange(precision):
	record_frame_path = "./output/cloudHistogram.npy"

	particle_frames = []

	vals = np.array([0.0, 1.0])

	corners = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1,3)
	frame = corners.astype(np.float32)
	particle_frames.append(frame)

	oklab_corners = OkLab.srgbToOklab(corners)
	frame = oklab_corners.astype(np.float32)
	particle_frames.append(frame)

	npy_frames = np.array(particle_frames, dtype=np.float32)
	np.save(record_frame_path, npy_frames)

	exp = 10**precision
	OKLAB_MIN = np.ceil(oklab_corners.min(axis=0) * exp) / exp 
	OKLAB_MAX = np.floor(oklab_corners.max(axis=0) * exp) / exp
	OKLAB_SIZE = OKLAB_MAX - OKLAB_MIN

	oklab_min_str = np.array2string(OKLAB_MIN, separator=', ')
	oklab_max_str = np.array2string(OKLAB_MAX, separator=', ')
	oklab_size_str = np.array2string(OKLAB_SIZE, separator=', ')

	print("OKLAB_BOX_MIN =   np.array(", oklab_min_str, ") # OkLab bounding box")
	print("OKLAB_BOX_MAX =   np.array(", oklab_max_str, ")")
	print("OKLAB_BOX_SIZE= np.array(", oklab_size_str, ")")


if __name__ == '__main__':
	np.set_printoptions(precision=8)
	color_depth = 256
	point_cloud = createMeshGrid(color_depth)

	precision=18
	calc_fallbackNorm(precision = precision)
	print("")
	calc_closestPoints(point_cloud, precision = precision)
	calc_OkLabVolume(point_cloud, precision = precision)
	print("")
	calc_OkLabRange(precision = precision)