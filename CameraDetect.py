# coding: utf-8
### hoge

import photos
import console
from objc_util import *
# added th
import deep_convnet
from PIL import Image, ImageOps
from image_mnist import image_mnist
import os
import shutil
import numpy as np
from mnist import load_mnist, load_thimage
import matplotlib.pyplot as plt


CIFilter, CIImage, CIContext, CIDetector, CIVector = map(ObjCClass, ['CIFilter', 'CIImage', 'CIContext', 'CIDetector', 'CIVector'])

def take_photo(filename='.temp.jpg'):
	img = photos.capture_image()
	if img:
		img.save(filename)
		return filename

def pick_photo(filename='.temp.jpg'):
	img = photos.pick_image()
	if img:
		img.save(filename)
		return filename

def load_ci_image(img_filename):
	data = NSData.dataWithContentsOfFile_(img_filename)
	if not data:
		raise IOError('Could not read file')
	ci_img = CIImage.imageWithData_(data)
	return ci_img

def find_faces(ci_img):
	opt = {'CIDetectorAccuracy': 'CIDetectorAccuracyHigh'}
	d = CIDetector.detectorOfType_context_options_('CIDetectorTypeFace', None, opt)
	faces = d.featuresInImage_(ci_img)
	return faces

def apply_perspective(corners, ci_img):
	tr, br, tl, bl = [CIVector.vectorWithX_Y_(c.x, c.y) for c in corners]
	filter = CIFilter.filterWithName_('CIPerspectiveCorrection')
	filter.setDefaults()
	filter.setValue_forKey_(ci_img, 'inputImage')
	filter.setValue_forKey_(tr, 'inputTopRight')
	filter.setValue_forKey_(tl, 'inputTopLeft')
	filter.setValue_forKey_(br, 'inputBottomRight')
	filter.setValue_forKey_(bl, 'inputBottomLeft')
	out_img = filter.valueForKey_('outputImage')
	return out_img

def write_output(out_ci_img, filename='.output.jpg'):
	ctx = CIContext.contextWithOptions_(None)
	cg_img = ctx.createCGImage_fromRect_(out_ci_img, out_ci_img.extent())
	ui_img = UIImage.imageWithCGImage_(cg_img)
	c.CGImageRelease.argtypes = [c_void_p]
	c.CGImageRelease.restype = None
	c.CGImageRelease(cg_img)
	c.UIImageJPEGRepresentation.argtypes = [c_void_p, CGFloat]
	c.UIImageJPEGRepresentation.restype = c_void_p
	data = ObjCInstance(c.UIImageJPEGRepresentation(ui_img.ptr, 0.75))
	data.writeToFile_atomically_(filename, True)
	return filename

def main():
	console.clear()
	i = console.alert('Info', 'This script detects faces in a photo.', 'Take Photo', 'Pick from Library')
	if i == 1:
		filename = take_photo()
	else:
		filename = pick_photo()
	if not filename:
		return
	ci_img = load_ci_image(filename)
	out_file = write_output(ci_img)
	console.show_image(out_file)
	# faces = find_faces(ci_img)
	### ここから ###
	mainfolder = "./"
	outputfolder = ""
	trainfolder = "training-images"
	testfolder = "test-images"
	if os.path.exists(trainfolder+'/1'):
		shutil.rmtree(trainfolder+'/1')
	###if os.path.exists(testfolder+'/1'):
	###	shutil.rmtree(testfolder+'/1')
	os.mkdir(trainfolder+'/1')
	###os.mkdir(testfolder+'/1')
	# convert from JPG to MNIST data format
	path_jpg = ".output.jpg"
	path_png = "output.png"
	img = Image.open(path_jpg)
	# アルファ値を考慮したグレイスケールに変換
	img = img.convert("L")
	# 白黒反転
	img = ImageOps.invert(img)
	img = img.point(lambda x: 0 if x < 130 else x)
	img = img.point(lambda x: (x * 1000))
	# 画像内で値が0でない最小領域を返す
	crop = img.split()[-1].getbbox()
	img.crop(crop)
	# 28x28のサイズに縮小
	img = img.resize((28, 28),Image.LANCZOS)
	img.save(path_png)
	# 下段に画像を表示
	ci_img = load_ci_image(path_png)
	out_file = write_output(ci_img,filename='./training-images/1/output'+'.png')
	console.show_image(out_file)
	# MNISTデータへの変換
	itm = image_mnist(main_folder = mainfolder, output_folder = outputfolder,
		train_folder = trainfolder, test_folder = testfolder)
	itm.image_to_mnist(toSquare = True, minSquareSize = 28, convertToGZip = True)
	# MNISTデータのロード
	###(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
	(x_train, t_train) = load_thimage(flatten=False, normalize=False)
	###print(x_train.shape)
	###print(x_train)
	print(x_train[0].shape)
	plt.imshow(x_train[0][0,:,:])
	plt.show()
	# DeepConvNetによる推測
	hoge = deep_convnet.DeepConvNet()
	hoge.load_params(file_name='deep_convnet_params.pkl')
	x = hoge.predict(x_train,train_flg=False)
	print(x.shape)
	print(x)
	# 最も確率の高い要素のインデックスを取得
	p = np.argmax(x)
	print('=== PREDICT ===')
	print(p)
	print('===============')

if __name__ == '__main__':
	main()

