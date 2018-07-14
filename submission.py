import helper
import numpy as np
import math
import copy
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## Developed by group SexyKitty (Bingxin Tong z /Yizhe Xing z5104614).
    with open(test_data, 'r') as infile:
            data_test=[line.strip().split(' ') for line in infile]

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()

    ## train parameters
    parameters={'gamma':'auto','C':1.0,'kernel':'linear','degree':3,'coef0':0.0}

    train_data = []
    for i in strategy_instance.class0:
        s = ''
        for j in range(0,len(i)):
            if j != 0:
                s += ' '
            s += str(i[j])
        train_data.append(s)
    for i in strategy_instance.class1:
        s = ''
        for j in range(0,len(i)):
            if j != 0:
                s += ' '
            s += str(i[j])
        train_data.append(s)
    ## Data processing 
    count_vect = CountVectorizer(tokenizer = lambda x:x.split())
    x_train_counts = count_vect.fit_transform(train_data)
    tfidf_transformer = TfidfTransformer()
    x_train = tfidf_transformer.fit_transform(x_train_counts)
    word_train = count_vect.vocabulary_
    class_0_train = np.zeros((len(strategy_instance.class0),1))
    class_1_train = np.ones((len(strategy_instance.class1),1))
    y_train = np.concatenate((class_0_train, class_1_train), axis = 0)
    ## Data trainning
    svm_clf = strategy_instance.train_svm(parameters, x_train, y_train.ravel())
    clf_coef = svm_clf.coef_.toarray()

    clf_coef_dict = {}
    for i in word_train:
        clf_coef_dict.update({i:clf_coef[0][word_train[i]]})
    clf_coef_sort = sorted(clf_coef_dict.items(),key=operator.itemgetter(1),reverse=False)
    ## test_data
    t = []
    for i in data_test:
        s = ''
        for j in range(0,len(i)):
            if j != 0:
                s += ' '
            s += str(i[j])
        t.append(s)

    x_test_counts = count_vect.transform(t)
    x_test = tfidf_transformer.transform(x_test_counts)

    label = svm_clf.predict(x_test)
    label_0 = 0
    label_1 = 0
    for i in label:
        if i == 0:
            label_0 += 1
        else:
            label_1 += 1

    ## modified data:del positive big, add negative big
    for i in data_test:
        modified_times = 20
        L = copy.deepcopy(i)
        del_list = []
        for word in L:
            if (word in clf_coef_dict) and ((word,clf_coef_dict[word]) not in del_list):
                del_list.append((word,clf_coef_dict[word]))
        del_list = sorted(del_list, key=lambda x:x[1],reverse=True)
        for word in del_list:
            if modified_times == 0:
                break
            if word[1] <= 0:
                break
            for j in L:
                if j == word[0]:
                    i.remove(word[0])
            modified_times -= 1

        # add
        for word in clf_coef_sort:
            if modified_times == 0:
                break
            if word[0] not in L:
                i.append(word[0])
                modified_times -= 1
                
    t_modify = []
    for i in data_test:
        s = ''
        for j in range(0,len(i)):
            if j != 0:
                s += ' '
            s += str(i[j])
        t_modify.append(s)

    x_modify_counts = count_vect.transform(t_modify)
    x_test_modify = tfidf_transformer.transform(x_modify_counts)

    label_modify = svm_clf.predict(x_test_modify)
    label_0 = 0
    label_1 = 0
    for i in label_modify:
        if i == 0:
            label_0 += 1
        else:
            label_1 += 1

    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data='./modified_data.txt'
    with open(modified_data, 'w') as modify_file:
        for x in t_modify:
            modify_file.write(x+'\n')
    ## You can check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## You are required to return the instance of this class.

## Test mode
##fool_classifier("test_data.txt")
