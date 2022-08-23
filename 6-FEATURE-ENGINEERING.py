####################
#FEATURE ENGINEERING
####################
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt 
import missingno as msno 
from datetime import date 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import LocalOutlierFactor 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data 

df = load_application_train()
df.head()

def load(): 
    data = pd.read_csv("datasets/titanic.csv")
    return data 

df = sns.load_dataset("titanic")
df.head()

############################
#1.Outliers(Aykırı Değerler)
############################

#Aykırı Değerleri Yakalama
    #Verinin tanımlayıcı ölçülerini yanıtlar. Görselleştirme olarak Box Plot(Kutu Grafiği), İstatiksel olarak, Z-score ve Max-Min Normalizasyonları

#Grafik Teknikleriyle Aykırı Değer Yakalama
sns.boxplot(x=df["age"])
plt.show()

#Aykırı Değerleri Nasıl Yakalanır?
q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)

iqr = q3 - q1 
up =q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["age"] < low) | (df["age"] > up)]
df[(df["age"] < low) | (df["age"] > up)].index

#Aykırı Değer var mı yok mu?
df[(df["age"] < low) | (df["age"] > up)].any(axis=None)
df[(df["age"] < low)].any(axis=None)

#1.Eşik değer belirledik. > 2.Aykırılara eriştik. > 3.Hızlıca aykırı değer var mı yok mu diye soruldu.

#İşlemleri Fonksiyonlaştırmak
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df,"age")
outlier_thresholds(df,"fare")

low,up = outlier_thresholds(df,"fare")

df[(df["fare"] < low) | (df["fare"] > up)].head()

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
check_outlier(df,"age")
check_outlier(df,"fare")

#grab_col_names
dff = load_application_train()
dff.head()

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
        car_th: int, optinal
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
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))

# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "age")

grab_outliers(df, "age", True)

age_index = grab_outliers(df, "age", True)


outlier_thresholds(df, "age")
check_outlier(df, "age")
grab_outliers(df, "age", True)

#Aykırı Değer Problemlerini Çözme
#1.Aykırı değerleri veri kümemizden çıkabiliriz.
#2.Aykırı değerlere yeni değerler atayarak sınırlayabiliriz.
#3.Onları analizimiz için zararsız bir değere dönüştürürüz.

#Silme
low, up = outlier_thresholds(df,"fare")
df.shape

df[~((df["fare"] < low) | (df["fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df,col)
    
df.shape[0] - new_df.shape[0]


#Baskılama Yöntemi (re-assignment with thresholds)
low, up = outlier_thresholds(df,"fare")

df[((df["fare"] < low) | (df["fare"] > up))]["fare"]

df.loc[((df["fare"] < low) | (df["fare"] > up)), "fare"]
df.loc[(df["fare"] > up),"fare"] = up
df.loc[(df["fare"] < low),"fare"] = low

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col,check_outlier(df,col))
    
for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col, check_outlier(df,col))
    
#RECAP , toparlamak.
df = load()
outlier_thresholds(df,"age")
check_outlier(df,"age")
grab_outliers(df,"age")

remove_outlier(df,"age").shape
replace_with_thresholds(df,"age")
check_outlier(df,"age")

#Çok Değişkenli Aykırı Değer Analizi:Local Outlier Factor
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df,col))
    
low, up = outlier_thresholds(df,"carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape 

low, up = outlier_thresholds(df,"depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape 

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50],style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape 

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#Missing Values(Eksik Değerler)
#1-Öncelikle veri setimizi inceleyelim.
#2-Sonra veri setimizde eksik veri olup olmadığını tespit edelim.
#3-Eksik verilerimizin hangi sebeplerden kaynaklandığına bakalım.
#4-Eksik verilerimizi veri setinden temizleyelim.
#5-Ekisk veriler yerine başka değerler ekleyelim.

df = load()
df.head()

#Eksik gozlem var mi yok mu sorgusu?
df.isnull().values.any()

#Değişkenlerdeki eksik değer sayısı?
df.isnull().sum()

#Değişkenlerdeki tam değer sayısı?
df.isnull().sum().sum()

#Veri setindeki toplam eksik değer sayisi?
df.isnull().sum().sum()

#En az bir tane ekisk değeri sahip olan gözlem birimleri?
df[df.isnull().any(axis=1)]

#Tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

#Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)],axis=1,keys=['n_miss','ratio'])
    print(missing_df, end="\n")
    
    if na_name:
        return na_columns
    
missing_values_table(df)
missing_values_table(df,True)

#Eksik Değer Problemini Çözme
missing_values_table(df)

#Çözüm1: Hızlıca Silmek
df.dropna().shape
#Çözüm2: Basit Atama Yöntemleri ile Doldurmak
df["age"].fillna(df["age"].mean()).isnull().sum()
df["age"].fillna(df["age"].median()).isnull().sum()
df["age"].fillna(0).isnull().sum()

#df.apply(lambda x: x.fillna(x.mean()),axis=0)
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "0" else x, axis=0).head()

dff.isnull().sum().sort_values(ascending=False)
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "0" and len(x.unique())<=10)else x, axis=0).isnull().sum()

#Kategorik Değişken Kırılımında Değer Atama

df.groupby("Sex")["Age"].mean()
df["Age"].mean()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"),"Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"),"Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull()
df.isnull().sum()

#Çözüm 3: Tahmine Dayalı Atama ile Doldurma
df = load()

#! cat_cols: Kategorik olanlar + Numerik gözüken ama kategorik olanlar - Kategorik gözüken ama kardinal olanlar.
#! num_cols: Tipi object olmayanlar(int-float)-Numerik gözüken ama kategorik olanlar.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

#Değişkenlerin Standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#KNN'ın Uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff),columns=dff.columns)
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(),["Age","age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#RECAP
df = load()

#Missing Table
missing_values_table(df)

#Sayısal değişkenleri direkt MEDİAN ile oluşturalım.
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "0" else x, axis=0).isnull().sum()

#Kategorik değişkenleri MODE ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "0" and len(x.unique()) <= 10) else x, axis=0)

#Kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#Tahmine Dayalı Atama ile Doldurma
#Gelişmiş Analizler

#Eksik Veri Yapısının İncelenmesi
"""Ne  güzeldir hazır verilerle çalışmak. Eksik gözlem derdin yok, ilişkisel veri tabanlarından verilerini birleştirme çaban yok
gürültülü veri desen ona ne hacet. En sevdiğimiz veri, hazır-işlem gerektirmeyen ve direkt analizlerimize başlayacağımız veri 
setleri olsa da hayatın gerçekleri var. Böyle verileri gerçek hayatta görmek çok güç."""

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi 
missing_values_table(df,True)
na_cols = missing_values_table(df,True)

def missing_vs_target(dataframe,target, na_columns):
    temp_df = dataframe.copy()
    
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(),1,0)
        
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count":temp_df.groupby(col)[target].count()}),end="\n\n\n")

missing_vs_target(df,"Survived",na_cols)

#RECAP 
df = load()
na_cols = missing_values_table(df,True)

#Sayısal Değişkenleri direkt median ile oldurmak
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "0" else x, axis=0).isnull().sum()

#Kategorik değişkenleri mode ile doldurma
df.apply(lambda x:x.fillna(x.mode()[0]) if (x.dtype == "0" and len(x.unique()) <= 10) else x, axis = 0).isnull().sum()

#Kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#Tahmine dayalı atama ile doldurma
missing_vs_target(df,"Survived", na_cols)

#3.Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)

#Label Encoding & Binary Encoding
"""
label Encoding: etiketlerin makine tarafından okunabilir biçime dönüştürülmesi için sayısal bir biçime dönüştürülmesi anlamına gelir. 
Makine öğrenimi algoritmaları daha sonra bu etiketlerin nasıl çalıştırılması gerektiğine daha iyi karar verebilir. 
Denetimli öğrenmede yapılandırılmış veri seti için önemli bir ön işleme adımıdır.
"""
df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"]) [0:5]
le.inverse_transform([0,1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df,col)
    
df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df,col)
    
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

#One-Hot Encoding
"""
Kategorik değişkenlerin ikili vektörler olarak temsil edilmesi, öncelikle kategorik değerlerin tamsayı değerlerle eşlenmesini gerektirir.
Daha sonra, her tam sayı değeri, 1 ile işaretlenmiş tam sayı indeksi dışında tamamı sıfır değer olan bir ikili vektör olarak temsil edilir.
"""
df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
pd.get_dummies(df,columns=["Sex","Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe,categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

#cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()
df.head()

#Rare Encoding
#1.Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
#2.Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
#3.Rare encoder yazalım.

#1.Kategorik değişkenlerin azlık - çokluk durumunun analiz edilmesi
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
        
for col in cat_cols:
    cat_summary(df, col)
    
#2.Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT":dataframe[col].value_counts(),
                            "RATIO":dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN":dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df,"TARGET",cat_cols)

#3.Rare encoder'ın yazılması
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == '0'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels),'Rare', temp_df[var])
     
    return temp_df

new_df = rare_encoder(df,0.01)

rare_analyser(new_df,"TARGET",cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#Feature Scaling(Özellik Ölçeklendirme)

#StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x-u) / s
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

#RobustScaler: Medyanı çıkar iqr'a böl.
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü 
    # X_Std = (X-X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col] 

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        
for col in age_cols:
    num_summary(df,col,plot=True)
    
#Numeric to Categorical: Sayısal değişlkenleri kategorik değişkenlere çevirme.
#Binning
df["Age_qcut"] = pd.qcut(df['Age'],5)

#Feature Extraction(Özellik Çıkarımı)

#Binary Features: Flag, Bool , True-False

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived":"mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat= %.4f, p-value = %.4f' % (test_stat, pvalue))

df.loc[((df['SibSp'] + df['Parch']) > 0),"NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch'])== 0),"NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived":"mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES","Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO","Survived"].sum()],
                                      
                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Text'ler üzerinden özellik türetmek.
df.head()

#Letter Count
df["NEW_NAME_COUNT"] = df["Name"].str.len()

#Word Count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

#Özel yapıları yakalamak
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived":["mean","count"]})

#Regex ile değişken türetmek
df.head()
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z] + \.)', expand=False)
df[["NEW_TITLE","Survived","Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age":["count","mean"]})

#Date Değişkenleri Üretmek
dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d")

#year
dff['year'] = dff['Timestamp'].dt.year

#month
dff['month'] = dff['Timestamp'].dt.month

#year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()
dff.head()

#Feature Interactions(Özellik Etkileşimleri)
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.groupby("NEW_SEX_CAT")["Survived"].mean()

#Titanic Uçtan Uca Feature Engineering & Data Preprocessing
df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#1.Feature Engıneerıng(Değişken Mühendisliği)
#Cabin Bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

#Name Count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

#Name Word Count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

#Name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

#Name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

#Family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

#age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

#age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

#sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#2.Outliers(Aykırı Değerler)
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#3.Missing Values(Eksik Değerler)

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#4.Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#5.Rare Encoding
rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)
df["NEW_TITLE"].value_counts()

#6.One-Hot Encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#7. Standart Scaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#8.Model
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Hiç bir işlem yapılmadan elde edilecek skor?
dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

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


plot_importance(rf_model, X_train)
