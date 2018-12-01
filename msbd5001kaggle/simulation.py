import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
import datetime

param = pd.read_csv('./data/test.csv')

res = []
for line in param.index[:]:
    # print(param.loc[line, :])
    param_class = param.loc[
        line, ['n_samples', 'n_features', 'n_classes', 'n_clusters_per_class', 'n_informative', 'flip_y', 'scale']]
    param_sgd = param.loc[line, ['penalty', 'l1_ratio', 'alpha', 'max_iter', 'random_state', 'n_jobs']]
    param_class_dct = param_class.to_dict()
    param_sgd_dct = param_sgd.to_dict()

    start_time = datetime.datetime.now().timestamp()
    x, y = make_classification(**param_class_dct)
    model = SGDClassifier(**param_sgd_dct)

    model.fit(x, y)
    end_time = datetime.datetime.now().timestamp()

    interval = (end_time - start_time)

    res.append(interval)
print(res)
df = pd.DataFrame({'id': {}, 'time': {}})
df.id = param.id
df.time = pd.Series(res)
df.to_csv('./data/gt.csv', index=0)