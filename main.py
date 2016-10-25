from svm import SVM

if __name__ == '__main__':
    svm = SVM('train_images/')
    svm.test('images/image10.png')
