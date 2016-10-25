from scipy import misc
import numpy as np

def to_numpy(imagepath):
    image = misc.imread(imagepath,True)
    image = image.astype(np.int64)
    # image.flatten()
#     print image.shape
#     print image
# to_numpy("black2.png")
    return image
def magic(input_data):
    result = []
    for char in input_data:
        result.append((char[0],to_numpy(char[1])))
    return result


