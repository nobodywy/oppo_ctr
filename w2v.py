import  jieba
import json
import gensim
from sklearn.externals import joblib

from urllib import request
from urllib.parse import quote

def vec(word):
    word = quote(word)
    url = 'http://219.223.189.238:5005/vec/'
    url += word
    with request.urlopen(url) as f:
        data = f.read()
        res =  data.decode('utf-8')
        try:
            re = json.loads(res)
            print()
        except:
            print('not found')
        print()

if __name__ == '__main__':
    vec('中国')
    #vec('sdasda')
    print()