#import gensim
#model = gensim.models.KeyedVectors.load_word2vec_format('./data/Tencent_AILab_ChineseEmbedding/fake.txt',binary=False)
###
#  分词 获得 用户词典
###

import  jieba
import json
import gensim
from sklearn.externals import joblib

def load_w2v(word_set):
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt',binary=False)
    a = word_set
    dict_ = {}
    word_wrong = 0
    for w in a:
        try:
            dict_[w] = model.get_vector(w)
        except:
            word_wrong += 1
            print('word wrong: ',word_wrong)
            continue
    f = open('new_w2v.txt','w',encoding='utf-8')
    l = len(dict_)

    f.writelines(str(l) + ' 200\n')
    for w in dict_:
        temp_a = dict_[w]
        str_a = str(temp_a[0])
        for i in range(1,200):
            str_a += ' ' + str(temp_a[i])
        f.writelines(w+' '+str_a+'\n')
    f.close()
    print('finish')

def read_row_data(filename):
    f = open(filename,encoding='utf-8')
    word_s = set()
    for line in f:
        l = line.split('\t')
        a = json.loads(l[1])
        for w in a:
            w_list = jieba.cut(w)
            for w_l in w_list:
                word_s.add(w_l)
    f.close()
    return word_s

if __name__ == '__main__':
    # f1 = './data/test.txt'
    # f2 = './data/train.txt'
    # f3 = './data/vail.txt'
    # word_test = read_row_data(f1)
    # word_train = read_row_data(f2)
    # word_vail = read_row_data(f3)
    # word_total = word_test.union(word_train)
    # word_total = word_total.union(word_vail)
    word_total = joblib.load('word_set.txt')
    load_w2v(word_total)
