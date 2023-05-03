import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


feature = pd.read_csv('feature.csv')
pca = PCA(n_components=5)
feature = feature.iloc[:, -5:]
min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1))
feature = min_max_normalizer.fit_transform(feature)
a = pca.fit(feature)
results = pca.transform(feature)
data = pd.DataFrame(results)

label = pd.read_csv('label.csv')
label = pd.DataFrame(label['label'])

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2020)
rfc = RandomForestClassifier()

# parameters = {'n_estimators': range(30, 100, 10), 'max_depth': range(3, 21, 2),
#               'min_samples_leaf': [5, 6, 7], 'max_features': [1, 2, 3, 4]}
#
# grid_rfc = GridSearchCV(rfc, parameters, scoring='f1_macro')

rfc.fit(X_train, y_train.values.ravel())
# print(grid_rfc.best_params_, grid_rfc.best_score_)

pred = rfc.predict(X_test)
print(metrics.classification_report(pred, y_test))


