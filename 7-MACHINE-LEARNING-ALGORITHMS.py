#######################################
#1-LİNEAR REGRESSİON DOGRUSAL REGRESYON
#######################################
#Sales Prediction with Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

pd.set_option('display.float_format', lambda x:'%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

#Simple Linear Regression with OLS Using Scikit-Learn
df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/advertising.csv")
df.head()
df.shape

X=df[["TV"]]
y=df[["sales"]]

#Model
reg_model = LinearRegression().fit(X, y)

#y_hat = b + w*TV

#sabit(b-bias)
reg_model.intercept_[0]

#TV'nin katsayısı (w1)
reg_model.coef_[0][0]

#Tahmin
#150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0]*150 

#500 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0]*500
df.describe().T

#Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws = {'color':'b','s':9}, ci=False,color="r")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()

#Tahmin Başarısı
#MSE
y_pred = reg_model.predict(X)
mean_squared_error(y,y_pred) #10.51
y.mean()
y.std()

#RMSE
np.sqrt(mean_squared_error(y,y_pred)) #3.24

#MAE
mean_absolute_error(y,y_pred) #2.54

#R-KARE
reg_model.score(X,y)

#Multiple Linear Regression
df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/advertising.csv")
X=df.drop('sales',axis=1)
y=df[["sales"]]

#Model
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)

y_test.shape
y_train.shape 

reg_model = LinearRegression().fit(X_train,y_train)

#sabit(b-bias)
reg_model.intercept_

#coefficients(w-weight)
reg_model.coef_

#TAHMİN
#Aşagıdaki gözlem değerlerine göre satışın beklenen değeri nedir?
#TV:30
#radio:10
#newspaper:40

#sabit(b-bias): 2.90794702
#coefficients(w-weight): 0.0468431,0.17854434,0.00258619

#Sales:2.90 + TV * 0.04 + radio * 0.17 + newspaper * 0.002
2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

#Tahmin Başarısını Değerlendirme

#Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred)) #1.73

#Train RKARE
reg_model.score(X_train,y_train)

#Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) #1.41

#Test RKARE
reg_model.score(X_test,y_test)

#10 katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error"))) #1.6913531708051797

#Simple Linear Regression with Gradient Descent from Scratch

#Cost Function MSE
def cost_function(Y,b,w,X):
    m=len(Y)
    sse=0
    
    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat-y)**2
        
    mse = sse / m
    return mse 

#Update_weights
def update_weights(Y,b,w,X,learning_rate):
    m=len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat-y)
        w_deriv_sum += (y_hat-y)*X[i]
    new_b = b-(learning_rate*1/m*b_deriv_sum)
    new_w = w-(learning_rate*1/m*w_deriv_sum)
    return new_b, new_w

#Train fonksiyonu 
def train(Y,initial_b,initial_w,X,learning_rate,num_iters):
    print("Starting gradient descent at b={0},w={1},mse={2}".format(initial_b,initial_w,
                                                                    cost_function(Y,initial_b,initial_w,X)))
    b=initial_b
    w=initial_w
    cost_history=[]
    
    for i in range(num_iters):
        b,w = update_weights(Y,b,w,X,learning_rate)
        mse = cost_function(Y,b,w,X)
        cost_history.append(mse)
        
        if i % 100 == 0:
            print("iter={:.d} b={:.2f} w={:.4f} mse={:.4}".format(i,b,w,mse))
            
            print("After{0} iterations b={1}, w={2}, mse={3}".format(num_iters,b,w,cost_function(Y,b,w,X)))
            return cost_history,b,w
        
df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/advertising.csv")

X=df["radio"]
Y=df["sales"]

#Hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history,b,w = train(Y,initial_b,initial_w,X,learning_rate,num_iters)



##################################################
#2-LOGİSTİC REGRESSİON(DOGRUSAL OLMAYAN REGRESYON)
##################################################
#Diabetes Prediction with Logistic Regression
    #İş Problemi
        #Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makina ögrenmesi modeli geliştirebilir misiniz?
        #Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitülerinde tutulan büyük veri setinin parçasıdır.
        #ABD'deki Arizona Eyaletinin en büyük 5.şehri olan Phonenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
        #768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır.
        #Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

    #Değişkenler:
        #Pregnancies:Hamilelik Sayısı
        #Glucose: Glikoz
        #BloodPressure: Kan Basıncı
        #SkinThickness: Cilt Kalınlığı
        #Insulın: İnsülin
        #BMI: Beden Kitle İndeksi
        #DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
        #Age: Yaş(yıl)
        #Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil(0)
        
    #İşlem Öncelikleri:
        #1.Exploratory Data Analysis(Keşifsel Veri Analizi)
        #2.Data Preprocessing(Veri Önİşleme)
        #3.Model & Prediction (Modelleme ve Tahmin Etme)
        #4.Model Evaluation(Model Doğrulama)
        #5.Model Validation:Holdout
        #6.Model Validation:10-Fold Cross Validation
        #7.Prediction for A New Observation
        
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,plot_roc_curve
from sklearn.model_selection import train_test_split,cross_validate

def outlier_thresholds(dataframe,col_name,q1=0.05,q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True 
    else:
        return False

def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] < up_limit),variable] = up_limit
    
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x:'%.3f'%x)
pd.set_option('display.width',500)

#Exploratory Data Analysis
df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/diabetes.csv")
df.head()
df.shape
#Target'ın Analizi
df["Outcome"].value_counts()

sns.countplot(x="Outcome",data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

#Feauture'ların Analizi
df.head()

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe,numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
    
for col in df.columns:
    plot_numerical_col(df,col)
    
cols = [col for col in df.columns if "Outcome" not in col]

df.describe().T

#Target vs Features
df.groupby("Outcome").agg({"Pregnancies":"mean"})

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")
    
for col in cols:
    target_summary_with_num(df,"Outcome",col)
    
#Data Preprocessing(Veri Ön İşleme)
df.shape
df.head()

df.isnull().sum()

df.describe().T    

for col in cols:
    print(col,check_outlier(df,col))
    
replace_with_thresholds(df,"Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])
    
df.head()

#Model & Prediction 
y = df["Outcome"]
X =df.drop(["Outcome"],axis=1)

log_model = LogisticRegression().fit(X,y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]

y[0:10]


#Model Evaluation
def plot_confusion_matrix(y,y_pred):
    acc = round(accuracy_score(y,y_pred),2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score:{0}'.format(acc),size=10)
    plt.show()
    
plot_confusion_matrix(y,y_pred)
print(classification_report(y, y_pred))

#Accuracy:0.78 , Precision:0.74 , Recall:0.58 , F1-Score: 0.65
#ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)  #0.83939

#Model Validation: Holdout
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.20,random_state=17)
log_model = LogisticRegression().fit(X_train,y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

#Accuary:0.78, Precision:0.74, Recall:0.58, F1-Score:0.65
#Accuary:0.77, Precision:0.79, Recall:0.53, F1-Score:0.63

plot_roc_curve(log_model,X_test,y_test)
plt.title('ROC Curve')
plt.plot([0,1],[0,1],'r--')
plt.show()

#AUC
roc_auc_score(y_test,y_prob)

#Model Validation: 10-Fold Cross Validation
y=df["Outcome"]
X= df.drop(["Outcome"],axis=1)

log_model = LogisticRegression().fix(X,y)
cv_results = cross_validate(log_model,
                            X,y,
                            cv=5,
                            scoring=["accuracy","precision","recall","f1","roc_auc"])
#Accuracy:0.78,Precision:0.74,Recall:0.58,F1-Score:0.65
#Accuracy:0.77,Precision:0.79,Recall:0.53,F1-Score:0.63

cv_results['test_accuracy'].mean() #Accuracy:0.7721
cv_results['test_precision'].mean() #Precision:0.7192
cv_results['test_recall'].mean() #Recall:0.5747
cv_results['test_f1'].mean() #F1-Score:0.6371
cv_results['test_roc_auc'].mean() #AUC:0.8327

#Prediction for A New Observation
X.columns

random_user=X.sample(1,random_state=45)
log_model.predict(random_user)

##############################################
#3-KNN K-NEAREST-NEIGHBOURS (K-EN YAKIN KOMŞU)
##############################################
#1-Exploratory Data Analysis (Keşifsel Veri Analizi)
#2-Data Preprocessing & Feature Engineering (Veri Önİşleme & Özellik Mühendisliği)
#3-Modeling & Prediction(Modelleme ve Tahmin)
#4-Model Evaluation(Model Doğrulama)
#5-Hyperparameter Optimization (Hiperparametreleri Optimizasyonu)
#6-Final Model ()

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

#1-Exploratory Data Analysis
df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/diabetes.csv")
df.head()
df.shape
df["Outcome"].value_counts()

#2-Data Preprocessing & Feature Engineering
y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled,columns=X.columns)

#3-Modeling & Prediction
knn_model = KNeighborsClassifier().fit(X,y)
random_user = X.sample(1,random_state=45)
knn_model.predict(random_user)

#4-Model Evaluation

#Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

#AUC için y_prob:
y_prob = knn_model.predict(X)

print(classification_report(y,y_pred)) #acc 0.83 , #f1 0.74 , #AUC
roc_auc_score(y,y_prob) #0.80

cv_results = cross_validate(knn_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()  #0.73 , 0.59, 0.78

#1.Örnek boyutu arttırılabilir.
#2.Veri Önişleme
#3.Özellik Mühendisliği
#4.İlgili algoritmalar için optimizasyon yapılır.

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors":range(2,50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X,y)
knn_gs_best.best_params_

#6-Final Model
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)
cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)

####################################################################
#4-Decision Tree Classfication:CART(SINIFLANDIRMA VE KARAR AĞAÇLARI)
####################################################################

#1.Exploratory Data Analysis
#2.Data Preprocessing & Feature Engineering
#3.Modeling using CART
#4.Hyperparameter Optimization with GridSearchCV
#5.Final Model
#6.Feature Importance
#7.Analyzing Model Complexity with Learning Curves 
#8-Vısualizing the Decision Rules
#9-Extracting Decision Rules
#10-Extracting Python / SQL / Excel Codes of Decision Rules
#11-Prediction using Python Codes
#12-Saving and Loading Model

#pip install pydotplus
#pip install skompiler
#pip install astor 
#pip install joblib

import warnings
import joblib 
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate,validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

warnings.simplerfilter(action='ignore',category=Warning)

#1-Exploratory Data Analysis
#2-Data Preprocessing & Feature Engineering
#3-Modeling using CART

df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X,y)

#Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

#AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:,1]

#Confusion matrix
print(classification_report(y,y_pred))

#AUC
roc_auc_score(y,y_prob)

#Holdout Yöntemi ile Başarı Değerlendirme
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=45)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train,y_train)

#Train Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:,1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train,y_prob)

#Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict(X_test)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)

#CV ile Başarı Değerlendirme
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X,y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
#0.7058568882098294
cv_results['test_f1']
#0.5710621194523633
cv_results['test_roc_auc'].mean()
#0.6719440950384347


#4-Hyperparameter Optimization With GridSearchCV
cart_model.get_params()

cart_params = {'max_depth':range(1,11),
               "min_samples_split":range(2,20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_
cart_best_grid.best_score_
random = X.sample(1,random_state=45)
cart_best_grid.predict(random)


#5-Final Model
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,random_state=17).fit(X, y)
cart_final.get_params()
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(cart_final,
                            X,y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1']
cv_results['test_roc_auc'].mean()


#6-Feature Importance
cart_final.feature_importances_ 

def plot_importance(model,features,num=len(X),save=False):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        
plot_importance(cart_final,X,num=5)


#7-Analyzing Model Complexity with Learning Curves(BONUS)
train_score, test_score = validation_curve(cart_final,X,y,
                                           param_name="max_depth",
                                           param_range=range(1,11),
                                           scoring="roc_auc",
                                           cv=10)

mean_train_score = np.mean(train_score,axis=1)
mean_test_score = np.mean(test_score,axis=1)

plt.plot(range(1,11),mean_train_score,
         label="Training Score",color='b')

plt.plot(range(1,11),mean_test_score,
         label="Validation Score",color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

def val_curve_params(model,X,y,param_name,param_range,scoring="roc_auc",cv=10):
    train_score,test_score = validation_curve(
        model,X=X,y=y,param_name=param_name,param_range=param_range,scoring=scoring,cv=cv)
    
    mean_train_score = np.mean(train_score,axis=1)
    mean_test_score = np.mean(test_score,axis=1)
    
    plt.plot(param_range,mean_train_score,
             label="Training Score",color='b')
    
    plt.plot(param_range,mean_test_score,
             label="Validation Score",color='g')
    
    plt.title(f"Validation Curve for {type(model)._name_}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
    
val_curve_params(cart_final,X,y,"max_depth",range(1,11),scoring="f1")
cart_val_params = [["max_depth",range(1,11)],["min_samples_split",range(2,20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model,X,y,cart_val_params[i][0],cart_val_params[i][1])

    
#8-Visualizing the Decision Tree

# conda install graphviz
# import graphviz

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")
cart_final.get_params()


#9-Extracting Decision Rules
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)


#10-Extracting Python Codes of Decision Rules
# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))
print(skompile(cart_final.predict).to('excel'))


#11-Prediction using Python Codes
def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns
x = [12, 13, 20, 23, 4, 55, 12, 7]
predict_with_rules(x)

x = [6, 148, 70, 35, 0, 30, 0.62, 50]
predict_with_rules(x)


#12-Saving and Loading Model
joblib.dump(cart_final, "cart_final.pkl")
cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]
cart_model_from_disc.predict(pd.DataFrame(x).T)



#################################################
#              GELİŞMİŞ AĞAÇ YÖNTEMLERİ
#Random Forests, GBM, XGBoost, LightGBM, CatBoost
#################################################

import warning 
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

warnings.simplefilter(action='ignore',category=Warning)

df = pd.read_csv("C:/Users/Monster/Desktop/Machine-Learning-Engineering-with-Python-main/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

#1-Random Forest (Rastgele Orman Ağaçları)
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model,X,y,cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

rf_params = {"max_depth":[5,8,None],
             "max_features":[3,5,7,"auto"],
             "min_samples_split":[2,5,8,15,20],
             "n_estimators":[100,200,500]}

rf_best_grid = GridSearchCV(rf_model,rf_params, cv=5, n_jobs=-1,verbose=True).fit(X,y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(rf_final,X,y,cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

def plot_importance(model,features,num=len(X),save=False):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",
                                                                   ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        
plot_importance(rf_final,X)

def val_curve_params(model,X,y,param_name,param_range,scoring="roc_auc",cv=10):
    train_score,test_score=validation_curve(
        model, X=X, y=y, param_name=param_name,param_range=param_range,scoring=scoring,cv=cv)
    
    mean_train_score = np.mean(train_score,axis=1)
    mean_test_score = np.mean(test_score,axis=1)
    
    plt.plot(param_range,mean_train_score,
             label="Training Score",color='b')
    
    plt.plot(param_range,mean_test_score,
             label="Validation Score",color='g')
    
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final,X,y,"max_depth",range(1,11),scoring="roc_auc")

#2-Gradient Boosting (GBM) Gradyan Arttırma
gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()

cv_results = cross_validate(gbm_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()  #0.7591715474068416
cv_results['test_f1'].mean() #0.634
cv_results['test_roc_auc'].mean() #0.82548

gbm_params = {"learning_rate":[0.01,0.1],
              "max_depth":[3,8,10],
              "n_estimators":[100,500,1000],
              "subsample":[1,0.5,0.7]}

gbm_best_grid = GridSearchCV(gbm_model,gbm_params,cv=5,n_jobs=-1,verbose=True).fix(X,y)
gbm_best_grid.best_params_
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17,).fit(X,y)

cv_results = cross_validate(gbm_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#3-XGBoost(EXTREME GRADİENT BOOSTİNG) Torba
xgboost_model = XGBClassifier(random_state=17,use_label_encoder=False)
xgboost_model.get_params()

cv_results = cross_validate(xgboost_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean() #0.75265
cv_results['test_f1'].mean() #0.631
cv_results['test_roc_auc'].mean() #0.7987

xgboost_params = {"learning_rate":[0.1,0.01],
                  "max_depth":[5,8],
                  "n_estimators":[100,500,1000],
                  "colsample_bytree":[0.7,1]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(xgboost_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#4-LightGBM
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results=cross_validate(lgbm_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate":[0.01,0.1],
               "n_estimators":[100,300,500,1000],
               "colsample_bytree":[0.5,0.7,1]}

lgbm_best_grid = GridSearchCV(lgbm_model,lgbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate":[0.01,0.02,0.05,0.1],
               "n_estimators":[200,300,350,400],
               "colsample_bytree":[0.9,0.8,1]}

lgbm_best_grid = GridSearchCV((lgbm_model), lgbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17,colsample_bytree=0.9,learning_rate=0.01)
lgbm_params = {"n_estimators":[200,400,1000,5000,8000,9000,10000]}

lgbm_best_grid = GridSearchCV(lgbm_model,lgbm_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17).fit(X,y)

cv_results=cross_validate(lgbm_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#5-CatBoost
catboost_model =CatBoostClassifier(random_state=17,verbose=False)

cv_results = cross_validate(catboost_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

catboost_params = {"iterations":[200,500],
                   "learning_rate":[0.01,0.1],
                   "depth":[3,6]}

catboost_best_grid = GridSearchCV(catboost_model,catboost_params,cv=5,n_jobs=-1,verbose=True).fit(X,y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(catboost_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#6-Feature Importance
def plot_importance(model,features,num=len(X),save=False):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",
                                                                   ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        
plot_importance(rf_final,X)
plot_importance(gbm_final,X)
plot_importance(xgboost_final,X)
plot_importance(lgbm_final,X)
plot_importance(catboost_final,X)

#7-Hyperparameter Optimization with RandomSearchCV (BONUS)
rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth":np.random.randint(5,50,10),
                    "max_features":[3,5,7,"auto","sqrt"],
                    "min_samples_split":np.random.randint(2,50,20),
                    "n_estimators":[int(x) for x in np.linspace(start=200,stop=1500,num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100, #denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X,y)
rf_random.best_params_
rf_random_final = rf_model.set_params(**rf_random.best_params_,random_state=17).fit(X,y)

cv_results = cross_validate(rf_random_final,X,y,cv=5,scoring=["accuracy","f1","roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#8-Analyzing Model Complexity with Learning Curves(BONUS)
def val_curve_params(model,X,y,param_name,param_range,scoring="roc_auc",cv=10):
    train_score,test_score = validation_curve(
        model,X=X,y=y,param_name=param_name,param_range=param_range,scoring=scoring,cv=cv)
    
    mean_train_score = np.mean(train_score,axis=1)
    mean_test_score = np.mean(test_score,axis=1)
    
    plt.plot(param_range,mean_train_score,label="Training Score",color='b')
    plt.plot(param_range,mean_test_score,label="Validation Score",color='g')
    
    plt.title("Validation Curve for{type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
    
rf_val_params = [["max_depth",[5,8,15,20,30,None]],
                 ["max_features",[3,5,7,"auto"]],
                 ["min_samples_split",[2,5,8,15,20]],
                 ["n_estimators",[10,50,100,200,500]]]

rf_model = RandomForestClassifier(random_state=17) 

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0],rf_val_params[i][1])
    
rf_val_params[0][1]


############################################################
#       DENETİMSİZ ÖGRENME (UNSUPERVISED LEARNING)         #
############################################################

#!pip install yellowbrick
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score,GridSearchCV

#1-K-Means (K-ortalamalar)
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/USArrests.csv",index_col=0)
df.head()    
df.isnull().sum()
df.info()
df.describe().T

sc=MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4,random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

#Optimum Küme Sayısının Belirlenilmesi
kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)
    
plt.plot(K,ssd,"bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans,k=(2,20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

#Final Cluster'ların Oluşturulması
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/USArrests.csv",index_col=0)
df["cluster"] = clusters_kmeans
df.head()

df["cluster"] = clusters_kmeans
df.head()
df["cluster"] = df["cluster"] + 1
df[df["cluster"]==5]
df.groupby("cluster").agg(["count","mean","median"])
df.to_csv("clusters.csv")

#2-Hierarchical Clustering
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/USArrests.csv",index_col=0)

sc=MinMaxScaler((0,1))
df = sc.fit_transform(df)

hc_average = linkage(df,"average")

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(7,5))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

#Kume Sayısını Belirlemek
plt.figure(figsize(7,5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5,color='r',linestyle='--')
plt.axhline(y=0.6,color='b',linestyle='--')
plt.show()

#Final Modeli Oluşturmak
from sklearn.cluster import AgglomerativeClustering 

cluster = AgglomerativeClustering(n_clusters=5,linkage="average")
clusters = cluster.fit_predict(df)

df=pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/USArrests.csv",index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = clusters_kmeans

#3-Principal Component Analysis
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/Hilters.csv")
df.head()
 
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary not in col"]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

#Optimum Bileşen Sayısı
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümelatif Varyans Oranı")
plt.show()

#Final PCA'ın Oluşturulması 
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

#-BONUS:Principal Component Regression
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/Hilters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes !="O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit,columns=["PC1","PC2","PC3"]).head()
df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit,columns=["PC1","PC2","PC3"]),
                      df[others]],axis=1)
final_df.head()

from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague","Division","League"]:
    label_encoder(final_df,col)
    
final_df.dropna(inplace=True)

y=final_df["Salary"]
X=final_df.drop(["Salary"],axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm,X,y,cv=5,scoring="neg_mean_squared_error")))
y.mean()

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(lm,X,y,cv=5,scoring="neg_mean_squared_error")))

cart_params = {'max_depth':range(1,11),
               "min_samples_split":range(2,20)}

#GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X,y)
cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_,random_state=17).fit(X,y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final,X,y,cv=5,scoring="neg_mean_squared_error")))

#BONUS:PCA ile Çok Boyutlu Veriyi İki Boyutta Göstermek
#Breast Cancer
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis","id"],axis=1)

def create_pca_df(X,y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit,columns=['PC1','PC2'])
    final_df = pd.concat([pca_df,pd.DataFrame(y)],axis=1)
    return final_df

pca_df = create_pca_df(X,y)

def plot_pca(dataframe,target):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1',fontsize=15)
    ax.set_ylabel('PC2',fontsize=15)
    ax.set_title(f'{target.capitalize()}',fontsize=20)
    
    targets = list(dataframe[target].unique())
    colors = random.sample(['r','b',"g","y"],len(targets))
    
    for t,color in zip(targets,colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices,'PC1'],dataframe.loc[indices,'PC2'],c=color,s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()
    
plot_pca(pca_df,"diagnosis")

#IRIS
import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"],axis=1)

pca_df = create_pca_df(X,y)
plot_pca(pca_df,"species")

#DIABETES
df = pd.read_csv("C:/Users/Monster/Desktop/MACHINE LEARNING SUMMER CAMP/3-machine_learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

pca_df = create_pca_df(X,y)
plot_pca(pca_df,"Outcome")
