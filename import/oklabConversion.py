import numpy as np

### Constants ###
OKLAB_8BIT_MARGIN = 1e-8
OKLAB_GAMUT_VOLUME = 0.054197416

def approxOkGap(point_count: int):
	return (OKLAB_GAMUT_VOLUME/max(1,point_count))**(1.0/3.0)

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
	r, g, b = lin[:,0], lin[:,1], lin[:,2]
	l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
	m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
	s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
	
	l_ = np.sign(l) * np.abs(l) ** (1/3)
	m_ = np.sign(m) * np.abs(m) ** (1/3)
	s_ = np.sign(s) * np.abs(s) ** (1/3)
	
	L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
	a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
	b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
	
	return np.stack([L,a,b], axis=1)

def oklabToLinear(lab: np.ndarray):
	L, a, b = lab[:,0], lab[:,1], lab[:,2]
	l_ = L + 0.3963377774*a + 0.2158037573*b
	m_ = L - 0.1055613458*a - 0.0638541728*b
	s_ = L - 0.0894841775*a - 1.2914855480*b
	
	l = l_**3
	m = m_**3
	s = s_**3
	
	r = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
	g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
	b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
	
	return np.stack([r,g,b], axis=1)

def srgbToOklab(col: np.ndarray):
	linRGB = srgbToLinear(col)
	oklab = linearToOklab(linRGB)
	return oklab

def oklabToSrgb(col: np.ndarray):
	linRGB = oklabToLinear(col)
	sRGB = linearToSrgb(linRGB)
	return sRGB
