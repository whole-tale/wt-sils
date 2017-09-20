#!/usr/bin/python

from __future__ import print_function
import glob
import Image, ImageMath
import os
import re

def colorToAlpha(inf, outf):
	os.system("gimp -i -b '(batch-set-alph-ix \"%s\" \"%s\")' -b '(gimp-quit 0)'" % (inf, outf))
	
def normalize(fn):
	colorToAlpha(fn, fn[:-5])

def notbw(fn):
	im = Image.open(fn)
	if im.mode == 'L':
		return False
	im2 = im.convert('HSV')
	h = im2.histogram()
	vol = float(im2.width * im2.height)
	smoment = 0.0
	for i in range(256, 512):
		smoment = smoment + h[i] * (i - 256) / vol

	grayscale = (smoment < 10)
	return not grayscale

def tooManyWords(bn):
	return len(bn.split()) > 1

lst = glob.glob('*.png.orig')

for fn in lst:
	print('%s...' % fn, end='')
	bn = fn[:-9]
	ok = False
	if len(bn) > 2 and bn[-2] == '_' and bn[-1].isdigit():
		print('DUP')
	elif notbw(fn):
		print('NOT BW')
	elif tooManyWords(bn):
		print('TOO COMPLEX')
	else:
		normalize(fn)
		os.remove(fn)
		ok = True
		print('OK')
	if not ok:
		os.remove(fn)
