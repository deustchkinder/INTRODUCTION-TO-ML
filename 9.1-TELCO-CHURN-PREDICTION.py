#####################################
#MÜŞTERİ TERK TAHMİN MODELİ OLUŞTURMA
#####################################

#İŞ PROBLEMİ:
    #Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
    
#VERİ SETİ HİKAYESİ:
    #Telco müşterikaybıverileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. 
    #Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.
"""
Hedef değişken “churn” olarak belirtilmiş olup; 1 müşterinin hizmeti terkettiğini, 0 ise terketmediğini göstermektedir.

CustomerId : Müşteri İd’si
Gender : Cinsiyet
SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
tenure : Müşterinin şirkette kaldığı ay sayısı
PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
TotalCharges : Müşteriden tahsil edilen toplam tutar
Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier 
from sklearn.svm import SVC 
import torch
import warnings 
warnings.filterwarning("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_rows',20)
pd.set_option('display.float_format',lambda x:'%.3f'%x)

from google.colab import drive
drive.mount('/content/drive')
path = ''
df = pd.read_csv(path)

"""
Her satırda tek bir müşteri bilgisi bulunmaktadır. Değişkenleri incelediğimizde demografik bilgileri, müşteriye sağlanan hizmetler ve fatura bilgilerinden oluşmaktadır.

* Müşteriye sağlanan hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
* Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler
* Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı
"""

df.head()
df.shape 

#GOREV1-KEŞİFÇİ VERİ ANALİZİ
    #Adım1-Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
    cat_cols, num_cols,cat_but_car = grab_col_names(df)
    cat_cols
    num_cols 
    cat_but_car 
#TotalCharges kardinal gözüküyor sürekli değişken olması gerekmektedir

#GOREV2-Gerekli düzenlemeleri yapınız(Tip hatası olan değişkenler gibi)
    """errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
        - If 'ignore', then invalid parsing will return the input.
    """
    
pd.to_numeric(df["TotalCharges"],errors='ignore').dtype
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')
#TotalCharges'ı değiştirdiğimiz için yeniden grab_col_names fonksiyonumuzu çağırıyoruz ve doğru gelip gelmediğini kontrol ediyoruz.

cat_cols,num_cols,cat_but_car = grab_col_names(df) 
num_cols 
#Hedef değişkenimizi de daha sonra dağılımları incelerken kullanabilmek adına Yes/No'dan 1/0 olarak değiştiriyoruz.

df["Churn"] = df["Churn"].apply(lambda x:1 if x == "Yes" else 0)

#GOREV3-Numerik ve Kategorik değişkenlerin veri içindeki dağılımını gözlemleyeniz.
#Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

#PhoneService değişkeninde %90'ı Yes geliyor. Hedef değişkenimiz Churn %73 terketmeyen %27 terkedenlerden oluşuyor. Bu model dengesizlik yaratır mı?
for col in cat_cols:
    cat_summary(df,col,plot=True)
    
#Numerik Değişkenlerin Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
        
#Tenure'a bakıldığında 1 aylık müşterilerin çok fazla olduğunu ardından da 70 aylık müşterilerin geldiği görülüyor.
#3 farklı sözleşme türümüz vardı. Bunun farklı sözleşmelerden kaynaklanıp kaynaklanmadığını inceleyebiliriz.
for col in num_cols:
    num_summary(df,col,plot=True)
    
#1 aylık ve iki yıl kontratlardan kaynaklı tenure içerisinde bir dağılım farklılığı varmış. Bunun aslında modelde etkili olmasını bekleriz. 
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show()

df[df["Contract"] == "Two Year"] ["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show()

df[df["Contract"] == "One Year"] ["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("One Year")
plt.show()

#GOREV4-Kategorik değişkenler ile hedef değişkenkler incelemesini yapınız?
def target_summary_with_cat(dataframe,target,categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count":dataframe[categorical_col].value_counts(),
                        "Ratio":dataframe[categorical_col].value_counts() / len(dataframe)}),end="\n\n\n")
    
for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

#Numerik değişkenlerin hedef değişkene göre analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)
    
#GOREV5-Aykırı Gözlem Analizi yapalım.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col,check_outlier(df,col))
    
#GOREV6-Eksik Gözlem Analizi yapalım.
#TotalCharges'taki boş gözlemlere bu noktada herhangi bir aksiyon almıyoruz.Daha sonra özellik çıkarımı 
df.isnull().sum().sort_values(ascending=False)

#BASE MODEL#
#Değişkenler üzerinde herhangi bir aksiyon almadan öne base bir model kurarak inceleyelim.
dff = df.copy()
cat_cols = [col for in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

dff.head()

y = dff["Churn"]
X = dff.drop(["Churn","customerID","TotalCharges"], axis=1)              
# TotalCharges içerisinde missing value olduğundan modeli çalıştırdığımızda anlamlı bir değişken olduğu için çalışmayacaktı, sokmadık

models = [('LR', LogisticRegression(random_state=46)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=46)),
          ('RF', RandomForestClassifier(random_state=46)),
          ('SVM', SVC(gamma='auto', random_state=46)),
          ('XGB', XGBClassifier(random_state=46)),
          ("LightGBM", LGBMClassifier(random_state=46))
          #,("CatBoost", CatBoostClassifier(verbose=False, random_state=46))
          ]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

#Gorev2 Feature Engineering 
#Bu noktada df dataframe'imize geri dönüyoruz.
df.head()
#Adım1-Eksik ve Aykırı gözlemler için gerekli işlemleri yapınız. Aykırı gözlemimiz yoktu onlarla ilgili aksiyon almamıza gerek yok. Eksik gözleme bu noktada aksiyon alabiliriz.
df["TotalCharges"].hist(bins=20)
plt.xlabel("TotalCharges")
plt.show()

df["TotalCharges"].fillna(df["TotalCharges"].median(),inplace=True)

#Adım2-Yeni değişkenler oluşturunuz.
#CustomerTime Değişkeni
#Müşterilik süresi ve terk arasındaki ilişkinin grafiksel incelenmesi
#Grafiksel olarak incelediğimizde özellikle ilk 12 aydaki terkin daha fazla olduğunu, müşterilik süresi arttıkça terk etme durumunun azaldığını yorumlayabiliriz. 
#Bu grafiği incelememiz sonucunda müşterilik aylarını yıl gruplarına çevirerek yeni bir değişken oluşturacağız.
    
bins = 50
plt.hist(df[df['Churn'] == 1].tenure,
         bins, alpha=0.5, density=True, label='Churned')
plt.hist(df[df['Churn'] == 0].tenure,
         bins, alpha=0.5, density=True, label="Didn't Churn")
plt.legend(loc='upper right')
plt.show()

df.loc[(df['tenure'] <=12),'CustomerTime'] = '1year'
df.loc[(df['tenure'] > 12) & (df['tenure'] <=48),'CustomerTime'] = '4years'
df.loc[(df['tenure'] > 48),'CustomerTime'] = '4years+'
df['CustomerTime'].value_counts()

#PaymentMethod_New değişkeni
#Ödeme yöntemlerinde ikisinin çek kullanımı ikisinin de banka ilişkili olmasından kaynaklı 4 grubu 2'ye indirdiğimiz bir değişken oluşturacağız.
df.loc[(df['PaymentMethod'] == 'Bank transfer (automatic)') | (df['PaymentMethod'] == 'Credit card (automatic)'), 'PaymentMethod_New'] = 'Bank'
df.loc[(df['PaymentMethod'] == 'Mailed check') | (df['PaymentMethod'] == 'Electronic check'), 'PaymentMethod_New'] = 'Check'

df['PaymentMethod_New'].value_counts()
df.drop('PaymentMethod',axis=1,inplace=True)

#LongTermContract Değişkeni
#Kontratı 1 veya 2 yıllık müşterileri daha uzun döneme sahip olduğu için onları bir grup olarak aldığımız bir değişken oluşturalım.

df["LongTermContract"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)
df['LongTermContract'].value_counts()

df.drop('Contract',axis=1,inplace=True)

#MaxPackageInternet Değişkeni
#Interneti varsa ve interneti olduktan sonra satın alabildiği diğer hizmetlerin de hepsine sahipse maximum paketi vardır diyebiliriz.
#InternetService OnliineSecurity OnlineBackup DeviceProtection TechSupport Streaming TV StreamingMovies

df['InternetService'].value_counts()

df.loc[:, "MaxPackageInternet"] = np.where((df["InternetService"] != 'No') & (df["OnlineSecurity"] == 'Yes')
                                               & (df["OnlineBackup"] == 'Yes') & (df["DeviceProtection"] == 'Yes')
                                               & (df["TechSupport"] == 'Yes') & (df["StreamingTV"] == 'Yes') & (df["StreamingMovies"] == 'Yes'), '1','0')

df['MaxPackageInternet'].value_counts()

#noSup Değişkeni 
#Yan hizmetlerimizden müşteriye destek amaçlı sağladığımız ürünlerin alınıp alınmadığını gösteren değişken oluşturalım.
df["noSup"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)
df['noSup'].value_counts()

#TotalServices Değişkeni
#Kişinin toplam aldığı servis sayısı gösteren değişken oluşturabiliriz.
df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

df['TotalServices'].value_counts()

#AvgPerMonth Değişkeni
#Müşterilerin ortalama aylık ne kadarlık ödeme yapmış olduğunu gösteren bir değişken oluşturabiliriz.
df["AvgPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

#CurrentIncrease Değişkeni
#Güncel fiyatın ortalama fiyata göre ne kadar daha fazla olduğunu görnek için değişken oluşturabiliriz.
df["CurrentIncrease"] = df["AvgPerMonth"] / df["MonthlyCharges"]

#StramingService Değişkeni 
#Yayın ve film akışı olup olmadığını gösteren değişkenlerden herhangi bir akış olup olmadığını gösteren değişken oluşturabiliriz.
df["StreamingService"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
df['StreamingService'].value_counts()

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Adım3-Encıoding İşlemleri gerçekleştirinix.
#LabelEncoding 
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)
    
#One Hot Encoding 
cat_cols
#Hedef değişkenimiz ve aslında grubu değil servis sayısını ifade eden TotalServices değişkenimizi dışarıda tutuyoruz
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "TotalServices"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

df.isnull().sum().sort_values(ascending=False)

#MODELLEME#
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

models = [('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=46)),
          ('RF', RandomForestClassifier(random_state=46)),
          ('SVM', SVC(gamma='auto', random_state=46)),
          ('XGB', XGBClassifier(random_state=46)),
          ("LightGBM", LGBMClassifier(random_state=46)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=46))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

    #Random Forest
rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8],
             "max_features": [3, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [100, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)   

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

    #XGBoost
xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.001],
                  "max_depth": [5, 8, 20],
                  "n_estimators": [100, 500],
                  "colsample_bytree": [0.5, 0.7]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)  

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

    #LightGBM
lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

    #Catboost
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

xgboost_best_grid.best_params_

from sklearn.metrics import confusion_matrix
y_pred = xgboost_final.predict(X)
print(confusion_matrix(y, y_pred))

cv_results['test_accuracy'].mean()

#Feature Importance#
def plot_importance(model, features, num=len(X), save=False):
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

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

