import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_excel('iris .xls')

y = data.Classification
x = data.drop('Classification', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, stratify = y, random_state = 52)

svc = SVC()
svc.fit(X_train,y_train)
pred_svc = svc.predict(X_test)
print(accuracy_score(y_test, pred_svc)*100)

pickle.dump(svc,open('model.pkl','wb') )