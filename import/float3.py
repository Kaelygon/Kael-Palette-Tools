##CC0 Kaelygon 2025
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional
from numba import njit



#Floating helpers
class float3:
	def __init__(self, x=0.0, y=0.0, z=0.0):
		self.f = (float(x), float(y), float(z))

	def __getitem__(self, i):
		return self.f[i]

	def __setitem__(self, i, value):
		self.f[i] = float(value)
	
#3D float operators
def sub_vec3(vec_a,vec_b):
	return float3(  vec_a[0] - vec_b[0], vec_a[1] - vec_b[1], vec_a[2] - vec_b[2] )
def add_vec3(vec_a,vec_b):
	return float3(  vec_a[0] + vec_b[0], vec_a[1] + vec_b[1], vec_a[2] + vec_b[2] )
def mul_vec3(vec_a,vec_b):
	return float3(  vec_a[0] * vec_b[0], vec_a[1] * vec_b[1], vec_a[2] * vec_b[2] )
def div_vec3(vec_a,vec_b):
	vec = float3( vec_a[0] / vec_b[0], vec_a[1] / vec_b[1], vec_a[2] / vec_b[2] )
	return vec

#Special 3D float operators
def lerp_vec3(vec_a,vec_b,a):
	return add_vec3( mul_vec3(vec_a, sub_vec3([1.0]*3,a) ), mul_vec3(vec_b,a) )
def pow_vec3(vec_a,vec_b):
	return float3(  vec_a[0] ** vec_b[0], vec_a[1] ** vec_b[1], vec_a[2] ** vec_b[2] )
def spow_vec3(vec_a,vec_b):
	return float3(  math_spow(vec_a[0],vec_b[0]),math_spow(vec_a[1],vec_b[1]),math_spow(vec_a[2],vec_b[2]) )
def sign_vec3(vec):
	return float3(  1 if c>=0 else -1 for c in vec )
def clip_vec3(vec,low,hi):
	return float3(  min(max(vec[0],low[0]),hi[0]), min(max(vec[1],low[1]),hi[1]), min(max(vec[2],low[2]),hi[2]) )
def lessThan_vec3(vec_a,vec_b):
	return float3(  vec_a[0]<vec_b[0], vec_a[1]<vec_b[1], vec_a[2]<vec_b[2] )
def roundAwayFromZero_vec3(vec):
    x = int(vec[0]) if vec[0] == int(vec[0]) else int(vec[0]) + (1 if vec[0] > 0 else -1)
    y = int(vec[1]) if vec[1] == int(vec[1]) else int(vec[1]) + (1 if vec[1] > 0 else -1)
    z = int(vec[2]) if vec[2] == int(vec[2]) else int(vec[2]) + (1 if vec[2] > 0 else -1)
    return (x, y, z)
def round_vec3(vec, digits):
	return float3( round(vec[0],digits),round(vec[1],digits),round(vec[2],digits) )
def valid_vec3(vec):
	for c in vec:
		if not math.isfinite(c):
			return False
	return True

#Vector operators
def dot_vec3(a, b):
	product = 0.0
	for i in range(3):
		product+= a[i] * b[i]
	return product

def cross_vec3(a, b):
	return float3(
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0]
  )

def length_vec3(vec):
	return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

def compSum_vec3(vec):
	return (vec[0] + vec[1] + vec[2])

#@njit
def lengthSq_vec3(vec):
	return (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

def norm_vec3(vec,eps=0.0):
	l = length_vec3(vec)
	if l<=eps:
		return float3( 1,0,0 )
	return float3( vec[0]/l, vec[1]/l, vec[2]/l )


### other math tools

#If negative base, return  sign(a) * abs(a)**b
MAX_FLOAT = 1.79e308
def math_spow(a:float,b:float):
	if abs(a)>1.0 and abs(b)>1.0 and (abs(a) > MAX_FLOAT ** (1.0 / max(b, 1.0))):
		return math.copysign(MAX_FLOAT, a)
	if a>0:
	   return a**b
	if b.is_integer():
		return -((-a)**b)
	return 0.0

def math_median(_list):
	if len(_list)==0:
		return None
	_list.sort()
	index=len(_list)//2
	if len(_list)%2==0:
		a = index
		b = max(index-1,0)
		return (_list[a] + _list[b]) / 2.0
	else:
		return _list[index]

def math_clip(a,lo,hi):
   return max(min(a,hi),lo)

def math_sign(a):
   return 1 if a>=0 else -1





def run_testFunction():
	a = float3(1.0, 2.0, 3.0)
	b = float3(2.0, 3.0, 4.0)
	c = mul_vec3(a, b)
	d = add_vec3(c,[1]*3)
	d=[3]*3

if __name__ == '__main__':
	run_testFunction()

 