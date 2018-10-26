import gensim
import json
from gevent import monkey
from flask import Flask
from gevent import pywsgi


monkey.patch_all()
app = Flask(__name__)

#model = gensim.models.KeyedVectors.load_word2vec_format('../data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt', binary=False)
model = gensim.models.KeyedVectors.load_word2vec_format('../data/Tencent_AILab_ChineseEmbedding/fake.txt', binary=False)
@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/sim/<word1>/<word2>')
def api_sim(word1,word2):
    try:
        sim = model.similarity(word1, word2)
        json_str = {}
        json_str['sim'] = sim
        js = json.dumps(json_str)
        return js
    except:
        return 'wrong word'


@app.route('/vec/<word>')
def api_vec(word):
    try:
        strin = model[word]
    except:
        return word + 'not found'
    strin = strin.tolist()
    json_str = {}
    json_str[word] = strin
    js = json.dumps(json_str)
    return js


if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=5005)
    server = pywsgi.WSGIServer(('0.0.0.0', 5005), app)
    server.serve_forever()