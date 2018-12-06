import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

raw_ntr = pd.read_csv('demo_ICFSR.csv')
raw_ntr.set_index(['STAT','Sample_Name'], inplace=True)

=========================================================================================================================================
# Define a function for dummy variable creating and train-test splitting
=========================================================================================================================================
# Convert data type
cvt_cat = lambda x: x.astype('category')

# non-default arguments followed by default ones
def preprocess_data(
    Dataset, Basic_label, Target_label, 
    Num_label=None, Cat_pase_label=None,
    random_state=21, test_size=0.3
    ):
    
    if Cat_pase_label is not None:
        Dataset[Cat_pase_label] = Dataset[Cat_pase_label].apply(cvt_cat, axis=0)
        pase_dummy = pd.get_dummies(Dataset[Cat_pase_label])
    else:
        pase_dummy = None
        
    if Num_label is not None:
        jointlabel = Basic_label + Num_label
    else:
        jointlabel = Basic_label
    
    X = Dataset[jointlabel]
    y = Dataset[Target_label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, test_size=test_size
        )

    # Data standardization fit X_train set and transform X_test set
    ss = StandardScaler()
    X_train_stdz = ss.fit_transform(X_train)
    X_test_stdz = ss.transform(X_test)
    X_train_numeric_df = pd.DataFrame(
        X_train_stdz, columns=X_train.columns, index=X_train.index
        )
    X_test_numeric_df = pd.DataFrame(
        X_test_stdz, columns=X_test.columns, index=X_test.index
        )    
    
    if pase_dummy is not None:
        X_train_analyse = pd.merge(
            X_train_numeric_df, pase_dummy, left_index=True, right_index=True
            )
        X_test_analyse = pd.merge(
            X_test_numeric_df, pase_dummy, left_index=True, right_index=True
            )
    else:
        X_train_analyse = X_train_numeric_df
        X_test_analyse = X_test_numeric_df
        
    return X_train_analyse, X_test_analyse, y_train, y_test
=========================================================================================================================================
# Create variable labels
Basic_info = ['Age', 'BMI']
Target = ['SHperPB']
Nutrients = raw_ntr.loc[:, 'Alcohol':'Cholesterol'].columns.tolist()
Num_pase_labels = ['walking', 'L_SPT', 'M_SPT', 'S_SPT', 'MSL_TRAIN']
Num_join_labels = Nutrients + Num_pase_labels
cat_labels = ['L_HW','H_HW','Repairs','Yard','Gardening','Caring']

# Create training and test sets
# With independent varialbes of Age, BMI, general PASE and nutrients
# Before running, 'PASE' should be included into Basic_info list
X_train, X_test, y_train, y_test = preprocess_data(
    Dataset=raw_ntr, Basic_label=Basic_info, 
    Nutrient_label=Nutrients, Target_label=Target)

# Create other training and test sets based on research purposes
# With independent varialbes of Age, BMI, subcategories of physical activity and nutrients
X_train, X_test, y_train, y_test = preprocess_data(
    Dataset=raw_ntr, Basic_label=Basic_info, 
    Num_label=Num_join_labels, Cat_pase_label=cat_labels,
    Target_label=Target)
=========================================================================================================================================
# Evaluate how will the fold of cross-validation affect training and prediction score
# Choose the n_folds with the best training and prediction score
=========================================================================================================================================
lasso = Lasso(random_state=0)
alphas = np.logspace(-3,1, num=40)
tuned_para = [{'alpha':alphas}]
n_folds = 6
alpha_bst = []
train_bst = []
prediction = []

for n in range(3,n_folds+1):
    clf = GridSearchCV(lasso, tuned_para, cv=n)
    clf.fit(X_train, y_train)
    alpha_bst.append(clf.best_estimator_.alpha)
    train_bst.append(clf.best_score_)
    prediction.append(clf.best_estimator_.score(X_test, y_test))
    
sum_list = np.column_stack((np.arange(3, n_folds+1), alpha_bst, train_bst, prediction))
df_sum = pd.DataFrame(
    sum_list, 
    columns=['fold number', 'best alpha', 'best train score', 'prediction score']
    )
=========================================================================================================================================
# Use the decided n_folds
n_folds = 3
clf_best = GridSearchCV(lasso, tuned_para, cv=n_folds)
clf_best.fit(X_train, y_train)
best_est = clf_best.best_estimator_

# Complete regression function
df_coef = pd.DataFrame()
df_coef['Feature Name']=  np.append(X_train.columns.values, 'intercept')
df_coef['coefficient'] = np.append(best_est.coef_, best_est.intercept_)
=========================================================================================================================================
# Plot hyperparameter tuning
scores = clf_best.cv_results_['mean_test_score']
scores_std = clf_best.cv_results_['std_test_score']
std_error = scores_std/np.sqrt(n_folds)
best_score = clf_best.best_score_
best_alpha = clf_best.best_estimator_.alpha

fig = plt.figure()
ax = fig.add_subplot(111)

ax.semilogx(alphas, scores)
ax.semilogx(alphas, scores+std_error, 'b--')
ax.semilogx(alphas, scores-std_error, 'b--')
ax.fill_between(alphas, scores+std_error, scores-std_error, alpha=0.3)
ax.axhline(best_score, linestyle='--', color='red')
ax.axvline(best_alpha, linestyle='--', color='red')

ax.set_xlabel('alpha')
ax.set_ylabel('CV mean_score +/- std error')
ax.annotate(
    '({:.3f},{:.3f})'.format(best_alpha,best_score),
    xy=(best_alpha,best_score),
    xytext=(best_alpha+0.3,best_score+0.5),
    arrowprops=dict(facecolor='black', shrink=0.01),
    fontsize=13
    )
ax.set_xlim(alphas[0], alphas[-1])
ax.set_title(
    "GridSearchCV for hyperparameter tunning of Lasso regression\n Fold number: {}".format(n_folds),
    fontsize=20
    )
plt.show()
