from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from models import lr, svc
from pandas import DataFrame

X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123)

mdls = [lr, svc]

for m in mdls:
    grid = GridSearchCV(m['model'], param_grid=m['grid'], return_train_score=True)
    grid.fit(X_train,y_train)
    print(DataFrame.from_dict(grid.cv_results_))

