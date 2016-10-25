import numpy as np
import os, glob
import pickle

from PIL import Image
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from skimage.filters import threshold_otsu

def classify_text(img_dir_name, clf_path):
    """
    Given the image directory path containing the images we want to predict and the path to the 
    classifier pickle, returns a list with tuples of the predicted character and the file path of
    the image. 
    """
    predict_img = load_predict_set(img_dir_name)
    
    with open(clf_path, 'rb') as clf_pkl:
        clf = pickle.load(clf_pkl)

    predict_data = []
    
    for file in predict_img:
        image = Image.open(file)
        predict_data.append(gray_image_to_array(image))
        
    predicted = clf.predict(np.asarray(predict_data).reshape(len(predict_data), -1))
    
    text_data = []
    
    for i in range(len(predicted)):
        text_data.append((predicted[i], predict_img[i]))
        
    return text_data

def train (train_set, label_set):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_set, label_set)
    return clf

def test(classifier, predict_set, expected_set):
    predicted = classifier.predict(predict_set)
    #print (predicted)
    #print (expected_set)
    count = 0
    for i in range(len(predicted)):
        print (predicted[i], expected_set[i], predicted[i] == expected_set[i])
        if (predicted[i] == expected_set[i]):
            count+= 1
    
    print ('precision: ' + str(count / len(predicted)))
    #print (metrics.classification_report(expected_set, predicted))
    
def load_predict_set(dir_name):
    new_predict_set = []
    
    file_list = os.listdir(dir_name)
    for file in file_list:
        new_predict_set.append(dir_name + '/' + file)
        
    return new_predict_set
    
def gray_image_to_array(img):
    img_array = np.asarray(img)
    binary = img_array > 0
    bin_array = binary.astype(int)
    
    return bin_array
    
def image_preprocess(img):
    img = img.resize((32, 32), Image.ANTIALIAS)
    rgb_data = np.asarray(img)
    gray_data = np.dot(rgb_data[...,:3], [0.299, 0.587, 0.114])
    thresh = threshold_otsu(gray_data)
    binary = gray_data > thresh
    threshed_data = binary.astype(int)
    
    return threshed_data
    
if __name__ == '__main__':
    train_set = []
    label_set = []
    predict_set = []
    expected_set = []
    
    for i in range(26):
        os.chdir('C:/Users/Justin/Desktop/Programming/College/CS_196/Project/EnglishImg/English/Img/GoodImg/Bmp/Sample0' + str(i + 11))
        count = 0
        for image in glob.glob('*.png'):
            img = Image.open(image)
            train_set.append(image_preprocess(img))
            label_set.append(chr(i + 65))
            count+=1
    
    for i in range(26):
        os.chdir('C:/Users/Justin/Desktop/Programming/College/CS_196/Project/EnglishImg/English/Img/GoodImg/Bmp/Sample0' + str(i + 37))
        for image in glob.glob('*.png'):
            img = Image.open(image)
            train_set.append(image_preprocess(img))
            label_set.append(chr(i + 97))
            
    print (len(train_set), len(train_set[0]), len(train_set[0][0]))
    print (len(label_set))
    
    data = np.asarray(train_set).reshape((len(train_set), -1))
    
    print (len(data), len(data[0]))
    
    clf = train(data, np.asarray(label_set))
    
    for j in range(26):
        os.chdir('C:/Users/Justin/Desktop/Programming/College/CS_196/Project/EnglishImg/English/Img/GoodImg/Bmp/Sample0' + str(j + 11))
        for file in glob.glob('*.png')[:41]:
            pimg = Image.open(file)
            predict_set.append(image_preprocess(pimg))
            expected_set.append(chr(j + 65))
    
    for j in range(26):
        os.chdir('C:/Users/Justin/Desktop/Programming/College/CS_196/Project/EnglishImg/English/Img/GoodImg/Bmp/Sample0' + str(j + 37))
        for file in glob.glob('*.png')[:41]:
            pimg = Image.open(file)
            predict_set.append(image_preprocess(pimg))
            expected_set.append(chr(j + 97))
        
    print (len(predict_set), len (expected_set))
    test_data = np.asarray(predict_set).reshape((len(predict_set), -1))
    
    test(clf, test_data, np.asarray(expected_set))
    
    with open('C:/Users/Justin/Desktop/Programming/College/CS_196/Project/text_classifier.pkl', 'wb') as clf_pkl:
        pickle.dump(clf, clf_pkl)
    