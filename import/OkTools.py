#OkTools.py
import numpy as np


### Color Conversion ###

def approxOkGap(point_count: int):
	return (OkTools.OKLAB_GAMUT_VOLUME/max(1,point_count))**(1.0/3.0)

def srgbToLinear(srgb: np.ndarray):
	cutoff = srgb <= 0.04045
	higher = ((srgb + 0.055) / 1.055) ** 2.4
	lower = srgb / 12.92
	return np.where(cutoff, lower, higher)

def linearToSrgb(lin: np.ndarray):
	lin = np.maximum(lin, 0.0)
	cutoff = lin <= 0.0031308
	higher = 1.055 * np.power(lin, 1/2.4) - 0.055
	lower = lin * 12.92
	return np.where(cutoff, lower, higher)

def linearToOklab(lin: np.ndarray):
	r, g, b = lin[...,0], lin[...,1], lin[...,2]
	l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
	m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
	s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
	
	l_ = np.sign(l) * np.abs(l) ** (1/3)
	m_ = np.sign(m) * np.abs(m) ** (1/3)
	s_ = np.sign(s) * np.abs(s) ** (1/3)
	
	L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
	a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
	b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
	
	return np.stack([L,a,b], axis=-1)

def oklabToLinear(lab: np.ndarray):
	L, a, b = lab[...,0], lab[...,1], lab[...,2]
	l_ = L + 0.3963377774*a + 0.2158037573*b
	m_ = L - 0.1055613458*a - 0.0638541728*b
	s_ = L - 0.0894841775*a - 1.2914855480*b
	
	l = l_**3
	m = m_**3
	s = s_**3
	
	r = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
	g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
	b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
	
	return np.stack([r,g,b], axis=-1)

def srgbToOklab(col: np.ndarray):
	linRGB = srgbToLinear(col)
	oklab = linearToOklab(linRGB)
	return oklab

def oklabToSrgb(col: np.ndarray):
	linRGB = oklabToLinear(col)
	sRGB = linearToSrgb(linRGB)
	return sRGB


#Manipulate arrays of colors
class OkTools:
	### Constants ###
	FALLBACK_NORM = np.array([ 0.57735027 ]*3) * [1,-1,1] #sqrt(1/3)

	OKLAB_8BIT_MARGIN =  7.011e-05  # minimum SRGB distance in oklab space
	OKLAB_GAMUT_VOLUME =  0.05356533  # (oklab gamut) / (srgb gamut)

	OKLAB_BOX_MIN =   np.array( [ 0.        , -0.23388758, -0.31152815] ) # OkLab bounding box
	OKLAB_BOX_MAX =   np.array( [1.        , 0.2745663 , 0.19856976] )
	OKLAB_BOX_SIZE = np.array( [1.        , 0.50845388, 0.51009792] )

	DARKEST_BLACK_LAB = srgbToOklab(np.array([[0.499/255,0.499/255,0.499/255]]))[0] #brighest 8-bit SRGB rounded to pure black 

	### Vec3 Tools ###

	@staticmethod
	def vec3Length(vector_list, axis=-1, keepdims=False):
		"""float[...,1] vec3Length(float[...,3] vector_list, int axis=1, bool keepdims=false)"""
		sq_sum = np.einsum('...i,...i->...', vector_list, vector_list)
		if keepdims:
			sq_sum = np.expand_dims(sq_sum, axis=axis)
		return np.sqrt(sq_sum)

	@staticmethod
	def vec3ArrayNorm(vector_list, axis=-1):
		"""float[...,3] vec3ArrayNorm(float[...,3] vector_list, int axis=1, bool keepdims=false)"""
		l = OkTools.vec3Length(vector_list, axis=axis, keepdims=True)
		norm = vector_list / (l + 1e-12)
		norm[np.all(vector_list == 0, axis=axis)] = OkTools.FALLBACK_NORM
		return norm, np.squeeze(l, axis=-1)


	### Color Tools ###

	@staticmethod
	def inOklabGamut(lab_list, eps = 1e-12, lower_bound = 0.0, upper_bound = 1.0, axis=-1):
		"""bool[] inOklabGamut(float[][3] lab_list, float eps = 1e-12, float lower_bound = 0.0, float upper_bound = 1.0 ))"""
		lin_list = oklabToLinear(lab_list)
		in_gamut = (lin_list > lower_bound-eps) & (lin_list < upper_bound+eps)
		in_gamut = in_gamut.all(axis=axis)
		return in_gamut

	@staticmethod
	def clipToOklabGamut(lab_list, eps = 1e-12, lower_bound = 0.0, upper_bound = 1.0, axis=-1):
		"""(float[][3] float[][3]) clipToOklabGamut(float[][3] lab_list, float eps = 1e-12, float lower_bound = 0.0, float upper_bound))"""
		lin_list = oklabToLinear(lab_list)
		out_gamut = (lin_list < lower_bound-eps) | (lin_list > upper_bound+eps)
		out_gamut = out_gamut.any(axis=axis)

		if not np.any(out_gamut):
			return lab_list, None

		new_pos = np.clip(lin_list[out_gamut],[0.0]*3,[1.0]*3)
		new_lab = linearToOklab(new_pos)

		clip_move = np.zeros_like(lab_list)
		clip_move[out_gamut] = new_lab - lab_list[out_gamut] #movement in ok space

		out_list = lab_list.copy()
		out_list[out_gamut] = new_lab
		return out_list, clip_move

	@staticmethod
	def calcChroma(lab_list):
		"""float[] calcChroma(float[][3] lab_list))"""
		return np.sqrt( lab_list[:,1]**2 + lab_list[:,2]**2 )

	@staticmethod
	def isOkSrgbGray(lab_list, threshold = 1.0/255.0):
		"""bool[] isOkSrgbGray(float[][3] lab_list, float threshold = 1.0/255.0))"""
		rgb_list = oklabToSrgb(lab_list)
		is_gray = (
			(abs(rgb_list[:,0]-rgb_list[:,1]) < threshold) & 
			(abs(rgb_list[:,1]-rgb_list[:,2]) < threshold)
		)
		return is_gray

	@staticmethod
	def srgbToHex(rgb):
		"""char* srgbToHex(float[3] rgb)"""
		rgb = np.clip(rgb,[0.0]*3,[1.0]*3)
		rgb = np.round(rgb * 255.0)	
		rgb = rgb.astype(np.uint8)
		return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])
