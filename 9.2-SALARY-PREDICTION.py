#########################
#BEYZBOL İLE MAAS TAHMINI
#########################

#Iş Problemi:
    #Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir makine
    #öğrenmesi projesi gerçekleştirilebilir mi?
    
#Veri Seti Hikayesi:
    #Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
    #Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
    #Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
    #1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
    #1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.
    
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

import warnings 
import pandas as pd 
import missingno as msno 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from helpers.data_prep import * 
from helpers.eda import * 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


df = pd.read_csv("C:/Users/Monster/Desktop/GİTHUB/datasets/hitters.csv")
df.head()

#EDA ANALIZI
df.describe()
check_df(df)

#Bağımlı değişkende 59 tane NA var!
#CAtBat, CHits outlier olabilir.

#BAĞIMLI DEĞİŞKEN ANALİZİ
import seaborn as sns
import matplotlib.pyplot as plt

df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()

sns.boxplot(df["Salary"])
plt.show()

#KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols

#KATEGORİK DEĞİŞKEN ANALİZİ
rare_analyser(df, "Salary", cat_cols)

#SAYISAL DEĞİŞKEN ANALİZİ
for col in num_cols:
    num_summary(df, col, plot=False)

#AYKIRI GÖZLEM ANALİZİ
for col in num_cols:
    print(col, check_outlier(df, col, q1=0.1, q3=0.9))

#1300 den sonraki değerleri veri setinden çıkartıyorum.
print(df.shape)
df = df[(df['Salary'] < 1350) | (df['Salary'].isnull())]  # Eksik değerleri de istiyoruz.
print(df.shape)
sns.distplot(df.Salary)
plt.show()

#AYKIRI DEĞERLERİ BASKILAMA
for col in num_cols:
    if check_outlier(df, col, q1=0.05, q3=0.95):
        replace_with_thresholds(df, col, q1=0.05, q3=0.95)

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.05, q3=0.95))

#EKSİK GÖZLEM ANALİZİ
missing_values_table(df)
# Salary bağımlı değişkeninde 59 Eksik Gözlem bulunmakta. Bunları çıkartmak bir çözüm yolu olabilir.

#KORELASYON ANALİZİ
import numpy as np
target_correlation_matrix(df, corr_th=0.3, target="Salary")
high_correlated_cols(df, plot=False, corr_th=0.90)

#VERİ ÖNİŞLEME

df['NEW_HitRatio'] = df['Hits'] / df['AtBat']
df['NEW_RunRatio'] = df['HmRun'] / df['Runs']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']

df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']
#Paydaya sıfır gelme ihtimaline karşılık paydadaki değişkenlere 1 eklenebilir.

#One Hot Encoder
df = one_hot_encoder(df, cat_cols, drop_first=True)

#ARA

#MODELLEME
df_null = df[df["Salary"].isnull()]  # Salary içerisindeki boş değerleri ayıralım.
df.dropna(inplace=True)  # Salarydeki eksik değerleri çıkartma

y = df['Salary']
X = df.drop("Salary", axis=1)

#HOLD OUT - MODEL VALIDATION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

##ROBUST SCALER:
#ALL DATA FINAL RMSE: 219.83899058361285
# cols = X.columns
# index = X.index
# from sklearn.preprocessing import RobustScaler
# transformer = RobustScaler().fit(X)
# X = transformer.transform(X)
# X = pd.DataFrame(X, columns=cols, index=index)

##STANDARD SCALER:
#ALL DATA FINAL RMSE: 186.16240421879607

# num_cols.remove("Salary")
# scaler = StandardScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])

#BASE MODELS
def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df

all_models = all_models(X, y, test_size=0.2, random_state=46, classification=False)

#RANDOM FORESTS MODEL TUNING
    #Tuning için hazırlanan parametreler. Tuning zaman aldığı için çıkan parametre değerlerini girdim.
rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

# rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1).fit(X_train , y_train)
# rf_cv_model.best_params_

best_params = {'max_depth': 10,
               'max_features': 8,
               'min_samples_split': 10,
               'n_estimators': 80}

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

#RANDOM FORESTS TUNED MODEL
rf_tuned = RandomForestRegressor(max_depth=10, max_features=4, n_estimators=150,
                                 min_samples_split=8, random_state=42).fit(X_train, y_train)

#TUNED MODEL TRAIN HATASI
y_pred = rf_tuned.predict(X_train)

print("RF Tuned Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred)))

#TUNED MODEL TEST HATASI
y_pred = rf_tuned.predict(X_test)
print("RF Tuned Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_tuned, X_train)

#Tuned edilmiş model nesnesinin kaydedilmesi
import pickle
pickle.dump(rf_tuned, open("rf_final_model.pkl", 'wb'))

#Tuned edilmiş model nesnesinin yüklenmesi
df_prep = pickle.load(open('rf_final_model.pkl', 'rb'))
