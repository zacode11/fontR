import math, requests, json, random
import pickle
from collections import OrderedDict
from wand.image import Image
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from numpy import genfromtxt, savetxt
import numpy as np
import pandas as pd
import matplotlib
import os, pickle
import operator
from PIL import Image

def intensity_from_rgb(red, green, blue):
    """
    Given RGB values for a given pixel, return intensity from 0 to 1
    """
    return 0.2989 * red + 0.5870 * green + 0.1140 * blue

def pixel_intensity_from_filename(filename):
    """
    Given a filename, returns the grayscale pixel intensity for every pixel.
    Input a any picture, will be scaled to 32x32. 
    """
    print("LOCATION: " +os.getcwd())
    im = Image.open(filename) #Can be many different formats.
    pix = im.load()
    width,height = im.size
    ratio = min(32.0/width, 32.0/height)
    
    new_im = Image.new("RGB", (32,32), "white")                
    thumb = im.resize((int(ratio * width),int(ratio * height)), Image.NEAREST)
    x_d = int((32 - thumb.size[0])/2)
    y_d = int((32 - thumb.size[1]) / 2)
    new_im.paste(thumb, (x_d,y_d,x_d + thumb.size[0],y_d+thumb.size[1]))
    
    width, height = new_im.size
    pix = new_im.load()
    
    a = np.zeros(shape=(width,height))
    frame = pd.DataFrame(a)
    for x in range(width):
        for y in range(height):
            frame[x][y] = int(round(intensity_from_rgb(pix[x,y][0], pix[x,y][1], pix[x,y][2])))
    return frame.values.flatten()

def pixel_array_from_snippet_list(char_list):
    """
    Input: list of tuples.
    row[0] == suspected character
    row[1] == filename of pic snippet
    Output: list of tuples
    row[0] == suspected characetr
    row[1] == 1x1024 intensity array
    """
    
    return_arr = []
    
    for row in char_list:
        return_arr.append((row[0], pixel_intensity_from_filename(row[1])))
    return return_arr

def get_font_probabs(char, pixels):
    """
    Input: suspected character, and a 1x1024 array of pixel intensities. 
    Output: 808x2 array of likelihoods of fonts, with [label, probability].
    Given: directory contains either .csv or .pkl for classifier to generate from for all letters.
    """
    classifier = None
    if os.path.isfile('pickles/' + char + '.pkl'):
        classifier = joblib.load("pickles/"+char+'.pkl')
    else:
        classifier = AdaBoostClassifier(n_estimators=100)
        #no header
        dataset = genfromtxt(char + '.csv', delimiter=',', dtype=None) 
        
        target = [x[0] for x in dataset]
        #train = [x[1:] for x in dataset]
        
        train = genfromtxt('a.csv', delimiter=',', dtype=float, usecols=range(1,1025))
        #given training data and labels. 
        #read from csv.
        print('training on ' + char + '.csv')
        classifier.fit(train, target)
        
        joblib.dump(classifier, char+'.pkl')
        
    labels = classifier.classes_
    probabilities = classifier.predict_proba(pixels)
    df = pd.DataFrame({'labels':labels, 'probabilities':probabilities[0]})

    return df

def get_font_list():
    """
    Return a list of all fonts in google fonts. 
    """
    r = requests.get('https://www.googleapis.com/webfonts/v1/webfonts?key=AIzaSyAfV0ZhaNyR3Wg4r9X6JdTYqsYkCD75jK0&sort=popularity')
    jon = json.loads(r.text)
    font_list = []
    for item in jon['items']:
        font_list.append(item['family'])
    return font_list

def get_font_dict():
    """
    Return a dictionary with each font being a key, and the value being 0.
    """
    return dict((key,0) for key in get_font_list())

def sigmoid(x):
    """
    Scale a value from 0 to 1, scaled according the sigmoid function. 
    Values over 0.5 are scaled up, under 0.5 scaled down.
    """
    return 1.0 / (1 + math.exp(-12 * (x - 0.5)))

def scale_probabilities(frame_list):
    """
    Given array ['a', ['label', 'f(label)'...]...]
    return ['a', ['label', 'Sigmoid(f(label))'...]...]
    """
    #given np.array [['font', p(font)]...]
    for index in range(len(frame_list)):
        for row in range(len(frame_list[index][1])):
            frame_list[index][1]['probabilities'][row] = sigmoid(float(frame_list[index][1]['probabilities'][row]))
    return frame_list

def sum_probabilities(frame_list):
    """
    Given array ['a', ['label', 'Sigmoid(f(label))'...]...]
    Return dict [{'label': sum('sigmoid(f(label))'))... }]
    """
    font_dict = get_font_dict()
    #uncomment following line to use simulated output.
    #font_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for row in frame_list:
        for index in range(len(row[1])):
            font_dict[row[1]['labels'][index].decode("utf-8")] += row[1]['probabilities'][index]
    return font_dict

def to_json(frame_dict):
    """
    Given a dictionary of {'label':'probability'...}, return json as specified earlier.
    """
    sorted_dict = OrderedDict(sorted(frame_dict.items(), key=operator.itemgetter(1), reverse=True))
    dict_list = []
    for key in sorted_dict:
        d = {}
        d['font']=key
        d['probability']=sorted_dict[key]
        #d[key] = sorted_dict[key]
        dict_list.append(d)
    return json.dumps({'data': dict_list})

def black_box(char_filename_list):
    """
    Input list [('a', 'filename')...]
    Return JSON specified as earlier. 
    """
    list_pixels = pixel_array_from_snippet_list(char_filename_list)
    for index in range(len(list_pixels)):
        list_pixels[index] = (list_pixels[index][0], get_font_probabs(list_pixels[index][0], list_pixels[index][1]))
    return to_json(sum_probabilities(scale_probabilities(list_pixels)))

