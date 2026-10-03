import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from wsn_ids.benchmark import load_data, benchmark


def test_invalid_data_and_duplicates(tmp_path):
    path = tmp_path/'data.csv'
    pd.DataFrame({'x':['bad']*8, 'label':['a']*4+['b']*4}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='numeric'):
        load_data(path,'label')
    pd.DataFrame({'id':range(9),'x':list(range(8))+[0],'label':['a']*4+['b']*4+['a']}).to_csv(path,index=False)
    x,y,names,removed = load_data(path,'label')
    assert len(y)==8 and removed==1 and names==['x']


def test_smote_preserves_holdout_and_runs_all_models():
    x,y = make_classification(n_samples=120,n_features=6,n_informative=4,weights=[.75,.25],random_state=2)
    y = y.astype(str)
    plain = benchmark(x,y,epochs=2)
    balanced = benchmark(x,y,smote=True,epochs=2)
    assert plain['test_index_sha256']==balanced['test_index_sha256']
    assert plain['test_rows']==balanced['test_rows']==30
    assert balanced['train_rows_after_smote']>balanced['train_rows_before_smote']
    assert set(balanced['models'])=={'decision_tree','logistic_regression','pytorch_mlp'}
    for value in balanced['models'].values():
        assert sum(map(sum,value['confusion_matrix']))==30
        assert 0<=value['report']['accuracy']<=1


def test_reject_empty_labels(tmp_path):
    path=tmp_path/'bad.csv'
    pd.DataFrame({'x':range(8),'label':['a']*4+[' ']*4}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Empty'):
        load_data(path,'label')
