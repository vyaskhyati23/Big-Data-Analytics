#!/usr/local/bin/python3
"""
preprocessor.py
Author: Khyati Vyas
Author: Satyajeet Shahane
Author: Siddharth Subramanian
This is a program cleans the data in the dataset.
"""
import sys
import codecs
import os
import re
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from os.path import dirname, abspath
import sklearn.model_selection
import sklearn.datasets
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.svm
from sklearn.metrics import confusion_matrix


def calSuccess(res, y_test):
    success = 0
    fail = 0
    i = 0
    for val in res:
        if val == y_test[i]:
            success = success + 1;
        else:
            fail = fail + 1;
        i = i + 1;

    print((success/i)*100)

def buildClassifier():
    p = os.path.dirname(os.path.realpath(__file__))
    p1 = os.path.join(p,'cleandata')
    files = sklearn.datasets.load_files(p1)   
    
    count_vector = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
    bow = count_vector.fit_transform(files.data)

    tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(bow)
    X = tfidf_transformer.transform(bow)    

    # X = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2), stop_words='english', use_idf='True').fit_transform(files.data)


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, files.target, test_size=0.8)
    
    # Multinomail NB
    # clf = sklearn.naive_bayes.MultinomialNB()
    # clf.fit(X_train, y_train)
    # res = clf.predict(X_test)
    # print("Naive bayes")
    # calSuccess(res, y_test)
    # print(confusion_matrix(y_test,res))

    #KNN
    # clf = sklearn.neighbors.KNeighborsClassifier(60, weights='distance')  
    # clf.fit(X_train, y_train)
    # res = clf.predict(X_test)
    # print("KNN")
    # calSuccess(res, y_test)
    # print(confusion_matrix(y_test,res))

    #SVM
    clf = sklearn.svm.LinearSVC()
    clf.fit(X_train, y_train)
    res = clf.predict(X_test)
    print("SVM Accuracy: ")
    calSuccess(res, y_test)
    print("SVM Confusion Matrix: ")
    print(confusion_matrix(y_test,res))


def rmCharLow(file):
    '''
    Function removes special characters and also converts the entire text to
    lowercase
    :param file: file in a newsgroup
    :return: cleaned file
    '''
    cleaned = []
    for line in file:
        cleanedLine = line.strip()
        if cleanedLine:
            linec = cleanedLine.lower()
            linec = re.sub('[^A-Za-z]+', ' ', linec)
            cleaned.append(linec)
    return cleaned


def rmWhitelines(file):
    '''
        Function removes white lines
        lowercase
        :param file: file in a newsgroup
        :return: cleaned file
        '''
    cleaned = []
    for line in file:
        if line:
            cleaned.append(line)
    return cleaned


def rmHeader(file):
    '''
        Function removes the header text from each file and returns only the
        body
        :param file: file in a newsgroup
        :return: cleaned file
        '''
    nLines = 0
    cleaned = []
    finished = False
    parts = file.split('\n')

    for part in parts:
        if part.startswith('Lines:'):
            val = part.split(' ')
            if val[1].isdigit() is False:
                nLines = 25
                break
            else:
                nLines = int(val[1])
                break
    cleaned = parts[-nLines:]
    return cleaned


def stemmer(file):
    '''
    Function which performs the stemming operation on the semi-cleaned data
    :param file: file in a newsgroup
    :return: cleaned file
    '''
    cleaned = []
    cleanedLine = ''
    ps = PorterStemmer()
    ss = SnowballStemmer('english')
    for line in file:
        words = word_tokenize(line)
        cleanedLine = ''
        for w in words:
            cleanedLine = cleanedLine + ' ' + ss.stem(w)
            # cleanedLine = cleanedLine + ' ' + ps.stem(w)
        cleaned.append(cleanedLine)
    return cleaned


def main():
    '''
    Main function which iterates through the entire dataset and performs
    necessary functions
    :return:
    '''

    fileData = ' '
    datapath = sys.argv[1]
    cleandir = os.path.join(os.getcwd(), 'cleandata')

    # path = dirname(dirname(abspath(__file__)))
    # datapath = os.path.join(path, 'datasmall')

    datapath = sys.argv[1]
    cleandir = os.path.join(os.getcwd(), 'cleandata')

    if not os.path.exists(cleandir):
         os.makedirs(cleandir)


    for subdir, dirs, files in os.walk(datapath):

        subdirname = subdir.split('/')
        subname = subdirname[-1]
        print("Entering subdirectory ", subname)
        cleansubdir = os.path.join(cleandir, subname)

        if not os.path.exists(cleansubdir):
             os.makedirs(cleansubdir)

        for file in files:
            filename = os.path.join(subdir, file)
            cleanedfil = '/' + file
            cleanedfile = cleansubdir + cleanedfil

            if file != '.DS_Store':
                with codecs.open(filename, "r", encoding='utf-8',
                                 errors='ignore') as fobj:

                    fileData = fobj.read()
                    fileData = rmHeader(fileData)
                    fileData = rmWhitelines(fileData)
                    fileData = rmCharLow(fileData)
                    fileData = stemmer(fileData)

                fobj2 = open(cleanedfile,"w")
                for line in fileData:
                    fobj2.write("%s\n" % line)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Data cleaned in %s seconds " % (time.time() - start_time))
    buildClassifier()
