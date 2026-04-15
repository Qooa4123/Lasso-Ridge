import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 定義評估函式
def get_metrics(model, X, Y, name):
    Y_pred = model.predict(X)
    metrics = {
        'Model': name,
        'Precision': precision_score(Y, Y_pred, zero_division=0),
        'Recall': recall_score(Y, Y_pred),
        'F1-score': f1_score(Y, Y_pred),
        'Selected_Vars': np.sum(model.named_steps['model'].coef_ != 0)
    }
    cm = confusion_matrix(Y, Y_pred)
    return metrics, cm



# 載入
df = pd.read_excel('final_data.xlsx')


# 設定應變量&自變量
Feature = ['1W_Rev', '1M_Rev', 'Mom_3M', 'Mom_6M', 'Mom_12M_1M', 'Vol_20D', 
            'Vol_60D', 'Vol_Down', 'Turnover', 'Turnover_Chg', 'Amihud', 'Vol_Z', 
            'Zero_Days', 'MA20_Bias', 'MA60_Bias', 'MACD', 'BB_Pos', 'RSI', 
            'ROE', 'Sales_Growth', 'Asset_Growth', 'Accruals']

X = df['1W_Rev', '1M_Rev', 'Mom_3M', 'Mom_6M', 'Mom_12M_1M', 'Vol_20D', 
            'Vol_60D', 'Vol_Down', 'Turnover', 'Turnover_Chg', 'Amihud', 'Vol_Z', 
            'Zero_Days', 'MA20_Bias', 'MA60_Bias', 'MACD', 'BB_Pos', 'RSI', 
            'ROE', 'Sales_Growth', 'Asset_Growth', 'Accruals']
Y = df['RR<0.1']


# 分割 S_train 與 S_test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)


#=============先用基本Lasso跑============

# 定義 CV 策略
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# Pipeline 普通 LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear'))
])


# GridSearchCV 切割與標準化(先用accuracy跟neg_log_loss評分)
param_grid = {'model__C': np.logspace(-4, 4, 20)}
grid_search_accuracy = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='accuracy')
grid_search_neg_log_loss = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='neg_log_loss')
grid_search_f1 = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='f1')

grid_search_accuracy.fit(X_train, Y_train)
grid_search_neg_log_loss.fit(X_train, Y_train)
grid_search_f1.fit(X_train, Y_train)


# 回報 Lasso 選中的變數
# 針對accuracy
best_pipeline_accuracy = grid_search_accuracy.best_estimator_
selected_coef_accuracy = best_pipeline_accuracy.named_steps['model'].coef_[0]
selected_vars_accuracy = [Features[i] for i, coef in enumerate(selected_coef_accuracy) if coef != 0]
print(f"Lasso Selected Variables by accuracy: {selected_vars_accuracy}")

#針對neg_log_loss
best_pipeline_neg_log_loss = grid_search_neg_log_loss.best_estimator_
selected_coef_neg_log_loss = best_pipeline_neg_log_loss.named_steps['model'].coef_[0]
selected_vars_neg_log_loss = [Features[i] for i, coef in enumerate(selected_coef_neg_log_loss) if coef != 0]
print(f"Lasso Selected Variables by neg_log_loss: {selected_vars_neg_log_loss}")

#針對f1
best_pipeline_f1 = grid_search_f1.best_estimator_
selected_coef_f1 = best_pipeline_f1.named_steps['model'].coef_[0]
selected_vars_f1 = [Features[i] for i, coef in enumerate(selected_coef_f1) if coef != 0]
print(f"Lasso Selected Variables by f1: {selected_vars_f1}")


#===========用scaled lasso and square-root lasso跑============

# 計算理論 Lambda，公式採簡化版，參考: lambda = c*sqrt(log(p)/n) , c 設為 1.1
n = X_train.shape[0]
p = X_train.shape[1]
c = 1.1
lambda_theory = c*np.sqrt(np.log(p) / n)
c_theory = 1 / lambda_theory

# 訓練理論模型
theory_lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear', C=c_theory))
])
theory_lasso_pipe.fit(X_train, Y_train)


#==============取得兩者表現==============
cv_metrics_accuracy, cv_cm_accuracy = get_metrics(best_pipeline_accuracy, X_test, Y_test, "Lasso_accuracy(CV)")
cv_metrics_neg_log_loss, cv_cm_y_neg_log_loss = get_metrics(best_pipeline_neg_log_loss, X_test, Y_test, "Lasso_neg_log_loss(CV)")
cv_metrics_f1, cv_cm_f1 = get_metrics(best_pipeline_f1, X_test, Y_test, "Lasso_f1(CV)")
theory_metrics, theory_cm = get_metrics(theory_lasso_pipe, X_test, Y_test, "Lasso (Theory)")



#============彙整結果表格==========
results_df = pd.DataFrame([cv_metrics_accuracy, cv_metrics_neg_log_loss, cv_metrics_f1, theory_metrics])
print(results_df)

# 顯示混淆矩陣
print("\nConfusion Matrix accuracy(CV):\n", cv_cm_accuracy)
print("\nConfusion Matrix neg_log_loss(CV):\n", cv_cm_y_neg_log_loss)
print("\nConfusion Matrix f1(CV):\n", cv_cm_f1)
print("\nConfusion Matrix (Theory):\n", theory_cm)