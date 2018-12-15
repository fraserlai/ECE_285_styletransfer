import numpy as np
import cv2

def get_mask(textfilepath, imgsize):
	with open(textfilepath, 'r') as f:
		polys = f.read()[:-1].split('\n')
		polys = [poly.split(', ') for poly in polys]
		polys = [np.array(poly, dtype=np.int32).reshape((-1, 1, 2)) for poly in polys]

	mask = np.zeros(imgsize, dtype=np.uint8)

	if len(polys) == 0:
		return mask

	cv2.fillPoly(mask, polys, color=255)
	mask = mask.astype(bool)
	# cv2.polylines(mask, polys, True, color=255, thickness=1)

	return mask

if __name__ == '__main__':
	img = cv2.imread('MCCALL_ROBINHOOD_T31_005.jpg')
	m = get_mask('MCCALL_ROBINHOOD_T31_005.txt', img.shape[:2])
	m = np.repeat(m[:, :, np.newaxis], 3, axis=2)
	temp = np.zeros(img.shape, dtype=np.uint8)
	temp[np.where(m == 1)] = img[np.where(m == 1)]
	cv2.imshow('image', cv2.resize(img, (0,0), fx=0.5, fy=0.5))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
