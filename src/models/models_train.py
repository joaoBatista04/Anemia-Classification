from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Parâmetros a serem testados no GridSearchCV
param_grids = {
    'KNN': {
        'model__n_neighbors': [3,5,7,9], 
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
    },
    'GaussianNB': {},
    'Random Forest': {
        'model__n_estimators': [100, 200, 300], 
        'model__max_depth': [None, 10, 20, 30, 40, 50],
        'model__criterion': ['gini', 'entropy'],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'model__n_estimators': [100, 200], 
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'model__gamma': [0, 0.1, 0.2]
    },
    'Decision Tree': {
        'model__max_depth': [None, 10, 20, 30, 40, 50],
        'model__criterion': ['gini', 'entropy'],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'LogisticRegression': {
        'model__multi_class': ['auto', 'multinomial', 'ovr'],
    },
    'AdaBoost': {
        'model__n_estimators': [50, 100, 150],
        'model__learning_rate': [0.01, 0.1, 1, 10]
    },
    'Gradient Boost': {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'model__gamma': ['scale', 'auto']
    },
    'MLP': {
        'model__activation': ['tanh', 'relu', 'sigmoid', 'softmax'],
        'model__alpha': [0.0001, 0.001, 0.01]
    }
}

#Modelos a serem testados no GridSearchCV
models = {
    'KNN' : KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(seed=42, nthread=1, verbosity=0),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(multi_class='auto', random_state=42),
    'Gradient Boost': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'MLP': MLPClassifier(random_state=42)
}

#Melhores hiperparâmetros encontrados no GridSearchCV e aplicados aos modelos
best_models = {
    'KNN' : KNeighborsClassifier(metric='manhattan', weights='distance', n_neighbors=3),
    'GaussianNB': GaussianNB(),
    'RandomForest': RandomForestClassifier(random_state=42, max_depth=20),
    'XGBoost': XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=None, colsample_bynode=None,
                               colsample_bytree=None, device=None,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=0, grow_policy=None,
                               importance_type=None,
                               interaction_constraints=None, learning_rate=0.1,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None,
                               monotone_constraints=None, multi_strategy=None,
                               n_estimators=100, n_jobs=None, nthread=1,
                               num_parallel_tree=None),
    'DecisionTree': DecisionTreeClassifier(random_state=42, min_samples_split=10, criterion='entropy', min_samples_leaf=1),
    'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000),
    'GradientBoost': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
    'AdaBoost': AdaBoostClassifier(learning_rate=0.1, random_state=42),
    'SVM': SVC(C=100, kernel='linear', probability=True, random_state=42),
    'MLP': MLPClassifier(activation='tanh', random_state=42)
}