The file classifier.py contains code which will clean data, build classifier(SVM), train it, test it
and give accuracy and confusion matrix. The script expects that the dataset should have the format 
similar to the 20 newsgroup data set. Pass the path of data dir to the script. This script uses 'punkt'
a tokenizer which needs to be installed using following steps- 

1. python3 (start the interpreter)
2. import nltk
3. nltk.download('punkt')


After installing the tokenizer, the main script can be invoked as sample invocation is -

python3 classifier.py ~tmh/pub/newsgroups


