from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd

train_data = pd.read_table('./data/train.txt',
        names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
val_data = pd.read_table('./data/vail.txt',
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)
test_data = pd.read_table('./data/test.txt',
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)
train_data = train_data[train_data['label'] != '音乐' ]
test_data['label'] = -1

train_data = pd.concat([train_data,val_data])
train_data['label'] = train_data['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']

for item in items:
    temp = train_data.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    train_data = pd.merge(train_data, temp, on=item, how='left')
    test_data = pd.merge(test_data, temp, on=item, how='left')
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train_data.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        train_data = pd.merge(train_data, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')
train_data_ = train_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
test_data_ = test_data.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)

print('train beginning')

X = np.array(train_data_.drop(['label'], axis = 1))
y = np.array(train_data_['label'])
X_test_ = np.array(test_data_.drop(['label'], axis = 1))
print('================================')
print(X.shape)
print(y.shape)
print('================================')

N = 5
xx_logloss = []
xx_submit = []
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)


for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
    fm = pylibfm.FM(
		num_factors=16,
		validation_size=0.05,
		num_iter=10
	)
    print('f1_score:',f1_score(y_test, np.where(fm.predict(X_test)>0.5, 1,0)))
    xx_logloss.append(fm.best_score['valid_0']['binary_logloss'])
    xx_submit.append(fm.predict(X_test_))


fm.fit(X,y)
a = fm.predict(X_test)






train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(X.toarray())
y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X,y)
a = fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
print('finish')