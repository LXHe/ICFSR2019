import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

raw_ntr = pd.read_csv('demo_ICFSR.csv')
raw_ntr.sort_index(inplace=True)
raw_dm = pd.get_dummies(raw_ntr, columns=['STAT'], drop_first=True)

X_labels = ['BMI', 'PASE', 'Protein', 'Se']
X = raw_dm[X_labels]
y = raw_dm['STAT_Sar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
===========================================================================================================================================
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

print(
    "Classification report:\n{}".format(
        classification_report(y_test, logreg.predict(X_test))
        )
    )
===========================================================================================================================================
# ROC plot
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC curve (area={:.2%})".format(auc))
ax.legend(loc='lower right')
ax.set_title("ROC curve with "+", ".join(X_labels))
plt.show()
