from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

lr = {
        'model':LogisticRegression()
        , 'grid':{
            'C':[1e-6, 1e-3, 1]
            ,'penalty':['l1', 'l2']
        }
    }


svc = {
        'model':SVC()
        , 'grid':{
            'C':[1e-6, 1e-3, 1]
        }
    }