#!/usr/bin/env python
#CC0 Kaelygon 2025
"""
Generate all dither variants
Some examples of good values
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palette.PalettizeImage import *

from PIL import ImageDraw, ImageFont

def addImgHeader(in_img, metadata, header_w):
	w, h = in_img.size

	gray_color = (128,128,128,255)
	header_img = Image.new("RGBA", (header_w, h), gray_color)
	draw = ImageDraw.Draw(header_img)
	font = ImageFont.load_default()
	draw.text((4, 4), metadata, fill=(0,0,0,255), font=font)

	out_img = Image.new("RGBA", (w + header_w, h), (0,0,0,0))
	out_img.paste(in_img, (0, 0))
	out_img.paste(header_img, (w, 0))
	return out_img

def generateExample(_input, _palette):
	
	output_path = "output"
	concat_output = output_path + "/" + "dither_variants.png"

	variant_fname = []
	method_list = ConvertPreset.DITHER_METHOD_KEYS
	for method in method_list:
		variant_fname.append(output_path+"/"+"variant_"+method+".png")

	preset_list = [
		ConvertPreset(
			image				= _input,
			palette			= _palette,
			output			= variant_fname[0],
			alpha_count		= 1,
			max_error		= 1.0,
			merge_radius	= 0.05,
			dither			= method_list[0],
		),
		ConvertPreset(
			image				= _input,
			palette			= _palette,
			output			= variant_fname[1],
			alpha_count		= 1,
			dither			= method_list[1],
			mask_size		= 16,
			mask_weight		= 1.0,
		),
		ConvertPreset(
			image				= _input,
			palette			= _palette,
			output			= variant_fname[2],
			alpha_count		= 1,
			dither			= method_list[2],
		),
		ConvertPreset(
			image				= _input,
			palette			= _palette,
			output			= variant_fname[3],
			alpha_count		= 1,
			dither			= method_list[3],
			mask_size		= 128,
			mask_weight		= 1.0,
		),
	]

	for preset in preset_list:
		Palettize_preset( preset )
		print("---")

	concat_img = None

	header_h = 16*7 #one line = 16px
	og_img = Image.open(_input).convert("RGBA")
	metadata = (
			"ORIGINAL"
	)
	og_img = addImgHeader(og_img, metadata, header_h)

	concat_img = np.array(og_img)

	for i, fname in enumerate(variant_fname):
		img = Image.open(fname).convert("RGBA")

		p=preset_list[i]
		metadata = (
			method_list[i].upper() + "\n" +
			"--max-error=" 	+ str(p.max_error) + "\n" +
			"--merge-radius="	+ str(p.merge_radius) + "\n" +
			"--dither=" 		+ str(p.dither) + "\n" +
			"--mask-size=" 	+ str(p.mask_size) + "\n" +
			"--mask-weight=" 	+ str(p.mask_weight) + "\n"
		)

		img = addImgHeader(img, metadata, header_h)

		if concat_img is None:
			concat_img = np.array(img)
		else:
			concat_img = np.concatenate((concat_img, np.array(img)), axis=0)

		os.remove(fname)

	concat_img = Image.fromarray(concat_img)
	concat_img.save(concat_output)

	print("generateExample() done!")


if __name__ == '__main__':
	img_path = "demoImages/LPlumocrista.png"
	tmp_crop = "output/LPlumocrista_crop.png"

	in_img = Image.open(img_path).convert("RGBA")
	crop_start = [66,215]
	crop_end = [crop_start[0]+192,crop_start[1]+192]
	in_img = in_img.crop( crop_start + crop_end )
	in_img.save(tmp_crop)

	generateExample(tmp_crop, "palettes/pal64.png")
	os.remove(tmp_crop)
