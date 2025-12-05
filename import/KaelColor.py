
import math
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional

from float3 import *

class KaelColor:
	TYPE_STR = ("SRGB","LINEAR","OKLAB")
	INVALID_COL = float3(1.0,0.0,0.5)

	def __init__(self, color_space: str, triplet: float3 = None, alpha: float = 1.0, fixed: bool = False):
		self.id = id(self)
		self.type = self.setType(color_space)
		self.col = triplet if triplet != None else self.INVALID_COL[:]
		self.alpha = float(alpha)
		self.fixed = fixed

	### Color conversion
	def _linearToSrgb(self, linRGB):
		cutoff = lessThan_vec3(linRGB, [0.0031308]*3)
		gammaAdj = spow_vec3(linRGB, [1.0/2.4]*3 )
		higher = sub_vec3( mul_vec3( [1.055]*3 , gammaAdj ), [0.055]*3 )
		lower = mul_vec3( linRGB, [12.92]*3 )

		return lerp_vec3(higher, lower, cutoff)

	def _srgbToLinear(self,sRGB):
		cutoff = lessThan_vec3(sRGB, [0.04045]*3)
		higher = spow_vec3(div_vec3(add_vec3(sRGB, [0.055]*3), [1.055]*3), [2.4]*3)
		lower = div_vec3( sRGB, [12.92]*3 )

		return lerp_vec3(higher, lower, cutoff)

	def _linearToOklab(self,col: float3):
		r,g,b = col
		l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
		m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
		s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

		l_ = math_spow(l, 1.0/3.0)
		m_ = math_spow(m, 1.0/3.0)
		s_ = math_spow(s, 1.0/3.0)

		return [
			0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
			1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
			0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_,
		]

	def _oklabToLinear(self, col: float3):
		L,a,b = col
		l_ = L + 0.3963377774 * a + 0.2158037573 * b
		m_ = L - 0.1055613458 * a - 0.0638541728 * b
		s_ = L - 0.0894841775 * a - 1.2914855480 * b

		l = l_*l_*l_
		m = m_*m_*m_
		s = s_*s_*s_

		return [
			+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
			-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
			-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
		]

	def _srgbToOklab(self):
		linRGB = self._srgbToLinear(self.col)
		oklab = self._linearToOklab(linRGB)
		return oklab

	def _oklabToSrgb(self):
		linRGB = self._oklabToLinear(self.col)
		sRGB = self._linearToSrgb(linRGB)
		return sRGB

	#### Public KaelColor funcitons

	#Returns an object copy instead of this instance
	def copy(self):
		new_col = KaelColor(self.type, list(self.col), self.alpha, self.fixed)
		return new_col

	def setType(self, color_space):
		color_space.upper()
		if not (color_space in self.TYPE_STR):
			print("Invalid type!")
			self.type = "SRGB"
		else:
			self.type = color_space
		return self.type

	#Return as different color space without mutating object
	def asSrgb(self):
		if self.type == "SRGB":
			return self.col
		if self.type == "OKLAB":
			return self._oklabToSrgb()
		if self.type == "LINEAR":
			return self._linearToSrgb(self.col)

	def asLinear(self):
		if self.type == "SRGB":
			return self._srgbToLinear(self.col)
		if self.type == "OKLAB":
			return self._oklabToLinear(self.col)
		if self.type == "LINEAR":
			return self.col

	def asOklab(self):
		if self.type == "SRGB":
			return self._srgbToOklab()
		if self.type == "OKLAB":
			return self.col
		if self.type == "LINEAR":
			return self._linearToOklab(self.col)

	#Mutates object .col value
	def toSrgb(self):
		self.col = self.asSrgb()
		self.type="SRGB"
		return self.col

	def toLinear(self):
		self.col = self.asLinear()
		self.type="LINEAR"
		return self.col

	def toOklab(self):
		self.col = self.asOklab()
		self.type="OKLAB"
		return self.col


	### Color tools ###

	def inOklabGamut(self, eps:float=1e-7):
		r_lin, g_lin, b_lin = self.asLinear()

		return (r_lin >= -eps) and (r_lin <= 1+eps) and \
				(g_lin >= -eps) and (g_lin <= 1+eps) and \
				(b_lin >= -eps) and (b_lin <= 1+eps)

	def clipToOklabGamut(self, calc_norm: bool=False, eps:float=1e-7):
		lin = self.asLinear()
		in_gamut =(
			(lin[0] >= -eps) and (lin[0] <= 1+eps) and
			(lin[1] >= -eps) and (lin[1] <= 1+eps) and
			(lin[2] >= -eps) and (lin[2] <= 1+eps)
		)
		if in_gamut:
			return [0.0,0.0,0.0]
	
		old_ok_pos = self.col
		self.setType("LINEAR")
		self.col = clip_vec3(lin,float3(),[1.0]*3)
		self.toOklab()
		return sub_vec3(self.col, old_ok_pos) #movement in ok space

	#Calc color values
	def calcChroma(self):
		ok = self.asOklab()
		return math.sqrt( ok[1]*ok[1] + ok[2]*ok[2] )

	def calcLum(self):
		return self.asOklab()[0]

	def isOkSrgbGray(self, threshold: float = 1.0/255.0):
		r,g,b = self.asSrgb()
		if( abs(r-g) < threshold and 
			abs(g-b) < threshold
		):
			return True
		return False

	def getSrgbHex(self):
		r,g,b = clip_vec3(self.asSrgb(),[0.0]*3,[1.0]*3)
		if not valid_vec3([r,g,b]):
			return "#00000000"
		r = int(round(r * 255.0))
		g = int(round(g * 255.0))
		b = int(round(b * 255.0))
		return "#{:02x}{:02x}{:02x}".format(r, g, b)
