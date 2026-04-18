import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 定義評估函式
def get_metrics(model, X, Y, name):
    Y_pred = model.predict(X)
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(Y, Y_pred),
        'Precision': precision_score(Y, Y_pred, zero_division=0),
        'Recall': recall_score(Y, Y_pred),
        'F1-score': f1_score(Y, Y_pred),
        'Selected_Vars': np.sum(model.named_steps['model'].coef_ != 0)
    }
    cm = confusion_matrix(Y, Y_pred)
    return metrics, cm

#定義0、1數量比例函式
def report_class_distribution(Y_train, Y_test):
    # 計算數量
    train_counts = Y_train.value_counts().sort_index()
    test_counts = Y_test.value_counts().sort_index()
    
    # 計算比例
    train_pct = Y_train.value_counts(normalize=True).sort_index() * 100
    test_pct = Y_test.value_counts(normalize=True).sort_index() * 100
    
    # 整理成表格
    dist_df = pd.DataFrame({
        'Train Count': train_counts,
        'Train %': train_pct,
        'Test Count': test_counts,
        'Test %': test_pct
    })
    return dist_df


# 載入
df = pd.read_excel(r'D:\派森\Lasso & Ridge\final_data.xlsx', sheet_name='工作表2')
df.columns = df.columns.str.strip()


# 設定應變量&自變量
Features = ['1W_Rev', '1M_Rev', 'Mom_3M', 'Mom_6M', 'Mom_12M_1M', 'Vol_20D', 
            'Vol_60D', 'Vol_Down', 'Turnover', 'Turnover_Chg', 'Amihud', 'Vol_Z', 
            'MA20_Bias', 'MA60_Bias', 'MACD', 'BB_Pos', 'RSI', 
            'ROE', 'Sales_Growth', 'Asset_Growth', 'Accruals']

X = df[Features]
Y = df['RR<0.1']


# 分割 S_train 與 S_test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)


#=============先用ridge跑============

# 定義 CV 策略
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# Pipeline 普通 LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l2', solver='lbfgs'))
])


# GridSearchCV 切割與標準化(用accuracy、neg_log_loss、f1評分)
param_grid = {'model__C': np.logspace(-4, 4, 20)}
grid_search_accuracy = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='accuracy')
grid_search_neg_log_loss = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='neg_log_loss')
grid_search_f1 = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring='f1')

grid_search_accuracy.fit(X_train, Y_train)
grid_search_neg_log_loss.fit(X_train, Y_train)
grid_search_f1.fit(X_train, Y_train)

#選出最佳模型
best_pipeline_accuracy = grid_search_accuracy.best_estimator_
best_pipeline_neg_log_loss = grid_search_neg_log_loss.best_estimator_
best_pipeline_f1 = grid_search_f1.best_estimator_


#==============取得表現==============
cv_metrics_accuracy, cv_cm_accuracy = get_metrics(best_pipeline_accuracy, X_test, Y_test, "Ridge_accuracy(CV)")
cv_metrics_neg_log_loss, cv_cm_y_neg_log_loss = get_metrics(best_pipeline_neg_log_loss, X_test, Y_test, "Ridge_neg_log_loss(CV)")
cv_metrics_f1, cv_cm_f1 = get_metrics(best_pipeline_f1, X_test, Y_test, "Ridge_f1(CV)")



#============彙整結果表格(Part 1 result)==========

results_df = pd.DataFrame([cv_metrics_accuracy, cv_metrics_neg_log_loss, cv_metrics_f1])
print(results_df)

# 顯示混淆矩陣
print("\nConfusion Matrix accuracy(CV):\n", cv_cm_accuracy)
print("\nConfusion Matrix neg_log_loss(CV):\n", cv_cm_y_neg_log_loss)
print("\nConfusion Matrix f1(CV):\n", cv_cm_f1)


# 抓取測試集的預測機率
probabilities_accuracy = best_pipeline_accuracy.predict_proba(X_test)[:, 1]
probabilities_neg_log_loss = best_pipeline_neg_log_loss.predict_proba(X_test)[:, 1]
probabilities_f1 = best_pipeline_f1.predict_proba(X_test)[:, 1]

# 將機率與實際標籤合併
prob_df = pd.DataFrame({
    'Actual_Y': Y_test.values,
    'Predicted_Prob_ac': probabilities_accuracy,
    'Predicted_Prob_nlog' : probabilities_neg_log_loss,
    'Predicted_Prob_f1' : probabilities_f1,
})


# 看機率的最大值、最小值、平均值
print("\n預測機率的統計描述：")
print(prob_df[['Predicted_Prob_ac',
    'Predicted_Prob_nlog',
    'Predicted_Prob_f1',]].describe().T)

#觀察模型截距與C值
result_df_C_intcpt = pd.DataFrame({
    'Model': ['Accuracy', 'Log-loss', 'F1'],
    'Intercept': [
        best_pipeline_accuracy.named_steps['model'].intercept_[0],
        best_pipeline_neg_log_loss.named_steps['model'].intercept_[0],
        best_pipeline_f1.named_steps['model'].intercept_[0],
    ],
    'C': [
        grid_search_accuracy.best_params_['model__C'],
        grid_search_neg_log_loss.best_params_['model__C'],
        grid_search_f1.best_params_['model__C'],
    ]
})
print("\n截距與C值：")
print(result_df_C_intcpt.set_index('Model').T)





#================第二部分，檢視樣本Y比例不均的問題，採三種方法解決=================

#先檢視訓練集跟測試集的Y(採分層抽樣，比例應該差不多)
distribution_data = report_class_distribution(Y_train, Y_test)
print(distribution_data)

#評分標準列表
scorers = ['accuracy', 'neg_log_loss', 'f1']

#Random Oversampling--------------------------------------------------------
# 建立過採樣器，會將少數類別增加到與多數類別數量一致
ros = RandomOverSampler(random_state=1)

# 僅對訓練集進行過採樣
X_train_ros, Y_train_ros = ros.fit_resample(X_train, Y_train)


# 建立一個 dictionary 來存放 ROS 的所有結果
ros_results = {}

for sc in scorers:
    #執行 GridSearchCV
    gs = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring=sc)
    gs.fit(X_train_ros, Y_train_ros)
    
    #提取最佳模型與參數
    best_model = gs.best_estimator_
    best_c = gs.best_params_['model__C']
    
    # 呼叫 get_metrics (產出 metrics 和 confusion matrix)
    # label 加上評分名稱方便辨識
    metrics, cm = get_metrics(best_model, X_test, Y_test, f"Lasso_ROS_{sc}")
    
    #打包存入字典
    ros_results[sc] = {
        'model': best_model,
        'metrics': metrics,
        'cm': cm,
        'best_C': best_c
    }


#SMOTE----------------------------------------------------------------------
#建立 SMOTE 採樣器
smote = SMOTE(random_state=42)

smote_results = {}

#重複在ros的動作
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

for sc in scorers:
    
    gs = GridSearchCV(pipe, param_grid, cv=cv_stratified, scoring=sc)
    gs.fit(X_train_smote, Y_train_smote)
    
   
    best_model = gs.best_estimator_
    best_c = gs.best_params_['model__C']
    
    
    metrics, cm = get_metrics(best_model, X_test, Y_test, f"Lasso_Smote_{sc}")
    
    
    smote_results[sc] = {
        'model': best_model,
        'metrics': metrics,
        'cm': cm,
        'best_C': best_c
    }


#Class-weight Adjustment------------------------------------------
#修改Pipeline：在 LogisticRegression 中加入 class_weight='balanced'
weighted_results = {}

pipe_weighted = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        penalty='l2', 
        solver='lbfgs', 
        class_weight='balanced', # <---(以樣本Y=1之比例進行加權)
        random_state=1
    ))
])


for sc in scorers:
    
    gs = GridSearchCV(pipe_weighted, param_grid, cv=cv_stratified, scoring=sc)
    gs.fit(X_train, Y_train)
    
    
    best_model = gs.best_estimator_
    best_c = gs.best_params_['model__C']
    
    
    metrics, cm = get_metrics(best_model, X_test, Y_test, f"Ridge_Weighted_{sc}")
    
    
    weighted_results[sc] = {
        'model': best_model,
        'metrics': metrics,
        'cm': cm,
        'best_C': best_c
    }


#==============印出調整過後的結果==============
def compile_summary_table(all_results_dicts):
    summary_data = []

    # 處理 CV 最佳化模型 (ROS, SMOTE, Weighted)
    for method_name, results_dict in all_results_dicts.items():
        for sc in ['accuracy', 'neg_log_loss', 'f1']:
            res = results_dict[sc]
            # 提取指標 
            m = res['metrics']
            # 計算選中變數
            coef = res['model'].named_steps['model'].coef_[0]
            num_vars = np.sum(coef != 0)

            summary_data.append({
                'Method': method_name,
                'Scoring': sc,
                'Best_C': res['best_C'],
                'Selected_Vars': num_vars,
                'Accuracy': m.get('accuracy', m.get('Accuracy', 0)),
                'Precision': m.get('precision', m.get('Precision', 0)),
                'Recall': m.get('recall', m.get('Recall', 0)),
                'F1-score': m.get('f1-score', m.get('F1-score', 0))
            })

    return pd.DataFrame(summary_data)

#彙整
all_dicts = {'ROS': ros_results, 'SMOTE': smote_results, 'Weighted': weighted_results}


summary_df = compile_summary_table(all_dicts)


print("\n" + "="*40 + " Exercise 2 綜合績效比較表 " + "="*40)
display(summary_df.sort_values(by=['Method', 'F1-score'], ascending=[True, False]))

#混淆矩陣==============================

def plot_all_cms(results_dict, method_name):
    # 設定畫布
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle(f'Confusion Matrices: {method_name} Method', fontsize=16, fontweight='bold')
    
    scorings = ['accuracy', 'neg_log_loss', 'f1']
    
    #畫出三個CV最佳化模型的CM
    for i, sc in enumerate(scorings):
        cm = results_dict[sc]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'CV Optimized ({sc})')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

#執行
plot_all_cms(ros_results, "ROS")
plot_all_cms(smote_results, "SMOTE")
plot_all_cms(weighted_results, "Weighted")