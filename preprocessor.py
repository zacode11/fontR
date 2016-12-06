import numpy as np
import cv2

class Preprocessor(object):

    def __init__(self, img_path):
        self._img = cv2.imread(img_path)
    
    def get_img(self):
        return self._img

    def get_contours(self):
        '''
        returns contours of img
        '''
        thresh_img = self.otsu_threshold()
        # don't change thresh_img, we need it below
        img_for_contour = thresh_img.copy()
        contours, hierarchy = cv2.findContours(img_for_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sub_imgs = []
        img_height, img_width = thresh_img.shape
        for contour in contours:
            # get rectangle contour is in
            x, y, w, h = cv2.boundingRect(contour)
            # if contours is too small it's probably garbage
            #if w * h < 0.0001 * (img_height * img_width):
                #continue
            sub_img = thresh_img[y:y+h, x:x+w]
            sub_img = cv2.resize(sub_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            sub_imgs.append([sub_img, x, y, w, h])
        return sub_imgs

    def otsu_threshold(self):
        '''
        Applies otsu's threshold to an image and returns the image
        '''
        gray_img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.fastNlMeansDenoising(gray_img, None, 10)
        ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY +
                                   cv2.THRESH_OTSU)
        return thresh_img
