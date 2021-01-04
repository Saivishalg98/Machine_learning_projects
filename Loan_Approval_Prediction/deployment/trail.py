# importing libraries for data manipulation
import numpy as np
import pandas as pd

# libraries for data split and model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score

# ROC -AUC score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# libraries for saving the files
import pickle

df = pd.read_csv('Predict_Loan_Approval_new.csv', index_col=0)
df = df[df['total_income'] < 25000]

# splitting the dataframe
X = df.loc[:, df.columns != 'Loan_Status']
y = df.loc[:, df.columns == 'Loan_Status']

X = pd.get_dummies(X, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)

# creating the instance of Logistic regression
log_reg_model = LogisticRegression(max_iter=6000)
log_reg_model = log_reg_model.fit(X_train.values, y_train.values.ravel())  # fitting the model over the training data
y_pred = log_reg_model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

f = open('loan_prediction.pkl', 'wb')
pickle.dump(log_reg_model, f)
f.close()