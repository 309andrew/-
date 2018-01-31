import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_validation import StratifiedKFold
import itertools
from sklearn import metrics
import xlsxwriter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")
'''
def AUC(X,y,lr):
    predictions =lr.predict_proba(X)
    y = label_binarize(y, classes=[3,4,5,6,7,8])
    false_positive_rate, recall, thresholds = roc_curve(y,predictions[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    #plt.show()
    return roc_auc
'''

def f_show(algo, feature_list):
    algo.feature_importances_.tolist()
    df = pd.DataFrame({'feature':feature_list, 
                       'feature_importance':algo.feature_importances_.tolist()})
    df = df.sort_values(by=['feature_importance'], ascending=False).reset_index(drop=True)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%f'%float(height),
                     ha='center', va='bottom')
    
    plt.style.use('ggplot')
    
    plt.figure(figsize=(20, 5))
    #plt.rcParams['font.family'] = 'DFKai-sb'  #標楷體
    gini = plt.bar(df.index, df['feature_importance'], align='center')
    plt.xlabel('Feature')  #X軸名稱
    plt.ylabel('Feature Importance')  #Y軸名稱
    plt.xticks(df.index, df['feature'])  #X軸項目名稱
    
    autolabel(gini)    
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def cnf_matrix_blue(x, y, algo, target_names):  #methon1
    plt.rcParams['font.family'] = 'DFKai-sb' #標楷體
    cnf_matrix = confusion_matrix(y, algo.predict(x))
    plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')
    plt.show()

# 建立Excel以便紀錄參數紀錄----------------------------
workbook = xlsxwriter.Workbook('output22222.xlsx')
worksheet = workbook.add_worksheet()

# 寫Excel的欄位 ----------------------------------------
i = 1
performance = ['Accuracies', 'Precision', 'Recall', 'f1_score', 'kappa']
for per in performance:
    worksheet.write(0,i, per) #excel A欄
    i+=1

# 變數宣告----------------------------------------------
#from sklearn.model_selection import StratifiedKFold
#cv = StratifiedKFold(n_splits=5)
cv = KFold(n_splits=5)
#cv = RepeatedKFold(n_splits=5, n_repeats=5)
#cv.get_n_splits(X, y)
#models, coefs, xytest= [], [], [[]]*cv.n_splits  # in case you want to inspect the models later, too
scores, precisions, recalls, f1_scores, AUCs, kappas= [], [], [], [], [], []
sco_all, pre_all, re_all, f1_all, AUC_all, kappa_all = [], [], [], [], [], []
X_train, y_train, X_test, y_test = [],[],[],[]
excel_x = 1
r = 0
criterion = ['gini', 'entropy']
weights = ['uniform', 'distance']
#kernel = ['linear', 'rbf', 'sigmoid']
kernel = ['rbf']
target_names = ['3','4','5','6','7','8'] 
sm = SMOTE(kind='borderline1')
best_score = 0

# 1.資料擷取 **************************************************************
df = pd.DataFrame.from_csv('winequality-red.csv', index_col=None)

# 2.資料清理 **************************************************************
df = df.fillna(df.mean()) #移除遺失值,補上平均數
df = df.drop_duplicates() #移除重複值
df = df.reset_index(drop=True) # 將index重設


# one-hot Encoding ------------------------------------------
'''
de = pd.get_dummies(df['department'])
sa = pd.get_dummies(df['salary'])
df = df.drop('department',axis=1)
df = df.drop('salary',axis=1)
df= df.join(de)
df= df.join(sa)
'''
# 設定X和y ---------------------------------------------------------------
X = df.drop(['quality'], axis=1)
#X = df[['alcohol',   # 酒精含量
#        'sulphates', # 硫酸鹽
#        'volatile acidity', # 揮發性酸度
#        'total sulfur dioxide' # 總二氧化硫
#        ]]

y = df['quality']
feature_list = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']
#標準化 Standardization ------------------------------------------
from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)

# 正規化 Normalization -----------------------------------------
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X = min_max_scaler.fit_transform(X)

# 取小數點後3位 ---------------------------------
for a in range(X.shape[0]):
    for b in range(X.shape[1]):
        X[a][b] = round(X[a][b], 3)

# 資料切分 ----------------------------------------------------------
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cv = KFold(n_splits=5)

# 將K-Fold 切出來的資料存到list，以便每個不同的分類器都是跑同樣的切分資料---------------------

for train, test in cv.split(X, y):
    #X_res, y_res = sm.fit_sample(X[train], y[train]) #對train資料做SMOTE
    #X_train.append(X_res) #將每一次K切出來的值，+到X_train的list中
    #y_train.append(y_res)
    X_train.append(X[train])
    y_train.append(y[train])
    X_test.append(X[test])
    y_test.append(y[test])

    
    
 # 3.資料分析 **************************************************************
# Overview of summary (quality V.S. quality)
quality_vs = df.groupby('quality')
quality_vs.mean()

# 決策樹畫特徵重要性 ----------------------------------------------
tree = RandomForestClassifier(max_features=None, max_depth = 5)
tree.fit(X, y)
f_show(tree, feature_list)

# 建立分類器 ----------------------------------------------
#parameter = [KNeighborsClassifier(n_neighbors=k, weights=wei) for wei in weights for k in range(5,50, 5)]
#parameter += [LogisticRegression(random_state=1)]
#parameter += [svm.SVC(C=c, kernel='rbf', gamma=ga, max_iter=5, probability=True) for c in range(1,20,1) for ker in kernel for ga in range(1,50,1) for it in range(1,501,100)]
#parameter += [DecisionTreeClassifier(criterion=cri, max_depth=k) for cri in criterion for k in range(4, 10, 1) ]
#parameter += [GaussianNB()]
#parameter += [SGDClassifier(loss="hinge", penalty="l2")]
#parameter += [Perceptron()]
parameter = [RandomForestClassifier(n_estimators=100, max_features='auto',oob_score=True, n_jobs=-1, random_state=9487941, warm_start=False)] #  for k in range(50, 130, 5) for fe in [1,5,'auto',None]
#parameter += [ExtraTreesClassifier(criterion=cri,n_estimators=k, max_depth=None, min_samples_split=2, max_features=fe) for k in range(50, 200, 25) for cri in criterion for fe in [1,5,'auto',None]]
#parameter += [AdaBoostClassifier(n_estimators=k) for k in range(10,30,10)]
#parameter += [GradientBoostingClassifier(n_estimators=n, learning_rate=1.0,max_depth=de, random_state=0) for n in range(50,1001,25) for de in range(1,10,4)]

# 集成學習，投票分類器-------------------------------------------------------------------
'''
clf1 = KNeighborsClassifier(n_neighbors=15, weights='distance')
clf2 = GradientBoostingClassifier(n_estimators=60, learning_rate=1.0, random_state=0)
clf3 = svm.SVC(decision_function_shape='ovo', kernel='rbf', gamma=1,C=17, probability=True)
clf4 = AdaBoostClassifier(n_estimators=55)
clf5 = RandomForestClassifier(criterion='gini', n_estimators=100)
clf6 = ExtraTreesClassifier(criterion='gini', n_estimators=150)

parameter += [VotingClassifier(estimators=[('knn', clf1), ('gbc', clf2), ('svc', clf3), ('ada', clf4), ('rfc', clf5), ('etc', clf6)], voting='hard')] #票票等值
parameter += [VotingClassifier(estimators=[('knn', clf1), ('gbc', clf2), ('svc', clf3), ('ada', clf4), ('rfc', clf5), ('etc', clf6)], voting='soft')] #票票不等值
'''
'''
# GridSearchCV 找voting最佳超參數 -------------------------------------------------------
from sklearn.grid_search import GridSearchCV
eclf = VotingClassifier(estimators=[ 
    ('knn', KNeighborsClassifier(weights='distance')),
    #('lr', LogisticRegression(LogisticRegression())),
    ('svm', svm.SVC(kernel='rbf', probability=True)),
    ('dtc', DecisionTreeClassifier()),
    ('gnb', GaussianNB()),
    #('sgd', SGDClassifier(loss="hinge", penalty="l2")),
    #('per', Perceptron()),
    ('rfc', RandomForestClassifier(oob_score=True)),
    ('etc', ExtraTreesClassifier()),
    ('gbc', GradientBoostingClassifier(learning_rate=1.0)),
    ('ada', AdaBoostClassifier()),
    ], voting='soft')

#Use the key for the classifier followed by __ and the attribute
params = {'knn__n_neighbors': np.arange(10,12,10), 'knn__weights':['uniform', 'distance'],
          #'lr__C': [c for c in range(1,20,2)], 
          'dtc__max_depth':np.arange(10,12,10), 'dtc__criterion':['gini','entropy'],
          'rfc__n_estimators':np.arange(10,12,10), 'rfc__max_features':[1,'auto','log2',None],
          'etc__n_estimators':np.arange(10,12,10), 'etc__max_features':[1,'auto','log2',None],
          'gbc__n_estimators':np.arange(10,12,10),
          'ada__n_estimators':np.arange(10,12,10),
          }
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=4)

grid.fit(X,y)

print (grid.best_params_)

'''
# 單一分類器-----------
'''
algo = RandomForestClassifier()
est = np.arange(10,101,30)
fea = [1,'auto', 'log2', None]
dep = [10,20,40,None]
params = {'n_estimators': est,'max_features':fea, 'max_depth':dep}
grid = GridSearchCV(estimator=algo, param_grid=params, cv=5)
grid = grid.fit(X_test[0], y_test[0])
print(grid.best_params_)
print(grid.best_score_)
algo = grid.best_estimator_
'''

    # 模型開始訓練-----------------------
for algo in parameter: 
    for kf in range(len(X_train)):        
        #X_res, y_res = sm.fit_sample(X_train[i], y_train[i])
        algo.fit(X_train[kf], y_train[kf])
        
        if type(algo) == KNeighborsClassifier: 
            keyword = ['KNN', algo.n_neighbors, algo.weights]
        elif type(algo) == LogisticRegression:
            keyword = ['LR', algo.penalty]
        elif type(algo) == svm.SVC: 
            keyword = ['SVC', algo.kernel, algo.gamma]
        elif type(algo) == DecisionTreeClassifier:
            keyword=['DTC', algo.criterion, algo.max_depth]
        elif type(algo) == RandomForestClassifier:
            keyword = ['RFC', algo.criterion, algo.n_estimators]
        elif type(algo) == ExtraTreesClassifier: 
            keyword = ['ETC', algo.n_estimators, algo.criterion, algo.verbose]
        keyword = algo 
        keyword = str(keyword)
        #keyword = keyword.replace('[', '')
        #keyword = keyword.replace(']', '')


        # 評估模型的正確率指標-------------
        score = algo.score(X_test[kf], y_test[kf])   
        precision = metrics.precision_score(y_test[kf], algo.predict(X_test[kf]), average='weighted')      
        
        recall = metrics.recall_score(y_test[kf], algo.predict(X_test[kf]), average='weighted')
        f1_score = metrics.f1_score(y_test[kf], algo.predict(X_test[kf]), average='weighted')        
        kappa = cohen_kappa_score(y_test[kf], algo.predict(X_test[kf]))
        #auc_1 = AUC(X_test[i], y_test[i], algo)
        #auc_1 = roc_auc_score(y_test[i], score, average='weighted')
        
        # 將分數存進list，等Kfold跑完算平均值------------
        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)   
        kappas.append(kappa)

        #AUCs.append(auc_1)               
        # ---- 跑完一個fold
        #coefs.append(algo.coef_)
        
    # 跑完全部的fold、1次模型--------------------------------------------------
    # 混淆矩陣 ----------------------------------------------------
    cnf_matrix_blue(X_test[kf], y_test[kf], algo, target_names)

    print(classification_report(y_train[kf], algo.predict(X_train[kf]), target_names=target_names))       
    print('----------------------------')
    print(classification_report(y_test[kf], algo.predict(X_test[kf]), target_names=target_names))       
    
    print(keyword)
    print('Accuracies:', np.mean(scores))
    sco_all.append(np.mean(scores))
    print('Precision:', np.mean(precisions))
    pre_all.append(np.mean(precisions))
    print('Recall:', np.mean(recalls))
    re_all.append(np.mean(recalls))
    print('f1_score:', np.mean(f1_scores))
    f1_all.append(np.mean(f1_scores))   
    print('kappa:', np.mean(kappas)) 
    kappa_all.append(np.mean(kappas))

    #print('AUC:', np.mean(AUCs))
    #AUC_all.append(np.mean(AUCs))
    if np.mean(scores) >= best_score:
        best_score = np.mean(scores)
        best_algo = algo
    print('\n')
    
    # 特徵重要性 ---------------------------------------------------------
    '''
    importances = algo.feature_importances_
    std = np.std([tree.feature_importances_ for tree in algo.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_list[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    '''
    # 將結果寫入Excel ----------------------------------------
    
    worksheet.write(excel_x, 0, keyword)
    worksheet.write(excel_x, 1, np.mean(scores))
    worksheet.write(excel_x, 2, np.mean(precisions))
    worksheet.write(excel_x, 3, np.mean(recalls))
    worksheet.write(excel_x, 4, np.mean(f1_scores))
    worksheet.write(excel_x, 5, np.mean(kappas))
    #worksheet.write(excel_x, 6, np.mean(AUCs))
    excel_x += 1
    
    s = 0
        
        
#初始化，以便計算下一個分類參數---------------------------------
    scores = []
    precisions = []
    recalls = []
    f1_scores = []              
    AUCs = []
    kappas = []
    print('-----------------------------------------------------')
'''
print(best_algo)
print('統計最大值')
print('Accuracies:', np.max(sco_all))
print('Precision:', np.max(pre_all))
print('Recall:', np.max(re_all))
print('f1_score:', np.max(f1_all))
print('kappa:', np.max(kappa_all))
#print('AUC:', np.max(AUC_all))
print('\n')
'''
# 關閉Excel 並將暫存的資料一次寫入Excel
workbook.close()
 

'''
import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(algo, f, pickle.HIGHEST_PROTOCOL)
    
with open('data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pickle_rfc = pickle.load(f)
 
#df = pd.read_csv('winequality-red.csv')
X2 = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y2 = df['quality']
sc = StandardScaler()
X2 = sc.fit_transform(X2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)
pickle_rfc.fit(X2_train, y2_train)
print('final report: \n',classification_report(y2_test, pickle_rfc.predict(X2_test), target_names=target_names))
print('final Accuracies:', pickle_rfc.score(X2_test, y2_test))
'''


'''
fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3,9):
        fpr[i], tpr[i], _ = roc_curve(y_test[kf][:, i], algo.predict_proba(X_test[kf])[:, i])
    
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test[kf].ravel(),  algo.predict_proba(X_test[kf]).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    
    plt.plot(fpr[3], tpr[3], color='darkorange',lw=2, label='ROC curve of class 3(area = %0.2f)' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], color='blue',lw=2, label='ROC curve of class 4(area = %0.2f)' % roc_auc[4])
    plt.plot(fpr[5], tpr[5], color='magenta',lw=2, label='ROC curve of class 5(area = %0.2f)' % roc_auc[5])
    plt.plot(fpr[6], tpr[6], color='green',lw=2, label='ROC curve of class 6(area = %0.2f)' % roc_auc[6])
    plt.plot(fpr[7], tpr[7], color='red',lw=2, label='ROC curve of class 7(area = %0.2f)' % roc_auc[7])
    plt.plot(fpr[8], tpr[8], color='yellow',lw=2, label='ROC curve of class 8(area = %0.2f)' % roc_auc[8])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
'''