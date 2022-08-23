#PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişme Fonksiyonel Keşifçi Veri Analizi(Advanced Functional Exploraty Data Analysis)
#! NumPy pandas'a bağlı bir kütüphanedir. Büyük Veri Setlerinde kullanacağız.
##########
# NUMPY 
##########
#Neden NumPy? (Why NumPy?)
#Numpy ile matematiksel işlemler yapılabilir ve bu işlemler, Python'un dahili dizilerini kullanarak mümkün olan oranla daha verimli ve daha az kodla yürütülür..
#Tek Boyulu Dizeler, Çok Boyutlu Dizeler, Matematiksel işlemlerdeki matrislerde kullanılır. Daha verimlidir. NumPy(Numeric Python)
#Verimli hafıza saklama ve operasyonel vektörellerde bize fayda sağlar, yüksek seviyeli işlemlerdendir. Meta bilgileri tutmak ve dizelerde çalışmayı sağlar. Daha az çaba ile daha fazla işlem yapmak.


#np isimlendirmesi kısayol ifadesiyle ilerleyen satırlarda da çağırabiliriz.

import numpy as np
a = [1,2,3,4]
b = [2,3,4,5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a * b

###############################################
#NumPy Array'ı Oluşturmak(Creating NumPy Arrays)
################################################
import numpy as np
np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))
np.zeros(10, dtype = int)
np.random.randint(0,10,size=10)
np.random.normal(10,4,(3,4))

####################################################
#NumPy Array Özellikleri (Attibutes of Numpy Arrays)
####################################################
import numpy as np
# ndim: Boyut Sayısı
# shape: Boyut Bilgisi
# size: Toplam Eleman Sayısı
# dtype: Array Veri Tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

##################################
#Yeniden Şekillendirme (ReShaping)
##################################
import numpy as np
np.random.randint(1,10, size=9)
np.random.randint(1,10, size=9).reshape(3,3)

##############################
#Index Seçimi(Index Selection)
##############################
import numpy as np
a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999 
a

import numpy as np
m = np.random.randint(10, size=(3,5))
m
m[0,0]
m[1,1]
m[2,3]
m[2,3]

m[2,3] = 999 
m[2,3] = 2.9

m[:,0]
m[1,:]
m[0:2,0:3]

###################################
#Fancy Index(Bir array ya da DataFrame içine bir indis listesi konularak o listedeki indislerin seçilmesidir.)
import numpy as np
v = np.arange(0,30,3)
v
v[1]
v[4]

catch = [1,2,3]
v[catch]

################################################
#Numpy'da Koşullu İşlemler (Conditions on Numpy)
################################################
import numpy as np
v = np.array([1,2,3,4,5])

#Klasik Döngü İle
ab = []
for i in v:
    if i < 3:
        ab.append(i)
  
#Numpy İle
v < 3
v[v < 3]
v[v > 3]
v[v == 3]
v[v >= 3]

################################################
#Matematiksel İşlemler (Mathematical Operations)
################################################
import numpy as np
v = np.array([1,2,3,4,5])

v / 5
v * 5  / 10
v ** 2
v - 1

np.subtract(v,1)
np.add(v,1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)

v = np.substract(v,1)

##########################################
#NumPy ile İki Bilinmeyenli Denklem Çözümü
##########################################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10
import numpy as np
a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a, b)

#########
#PANDAS
#########

#Pandas Series
#Veri Okuma(Reading Data)
#Veriye Hızlı Bakış(Quick Look at Data)
#Pandas'ta Seçim İşlemleri(Selection in Pandas)
#Toplulaştırma ve Gruplama(Aggregation & Grouping)
#Apply ve Lambda
#Birleştirme(Join) İşlemleri

##############
#Pandas Series
##############
import pandas as pd 
s = pd.Series([10,77,12,4,5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

#########################
#Veri Okuma(Reading Data)
#########################
import pandas as pd
df = pd.read_csv("C:/Users/Monster/Desktop/pythonProgramlama (1)/pythonProgramlama/python_for_data_science/data_analysis_with_python/datasets/advertising.csv")
df.head()  #Pandas cheatsheet.

#########################################
#Veriye Hızlı Bakış (Quick Look at Data)
#########################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head() #CSV Veri Okuma
df.tail() #CSV Son satırı kullanma
df.shape #CSV satır ve sutun sayısını kümeleme 
df.info() #CSV indeks numarasına göre sutunları gruplama(index-sutun-boş olmayan adet-veri tipi) verinin metadatalarına getirebilir.
df.columns #CSV sutunları listeleme
df.index #CSV indeks aralığı(başlangıç,bitiş,aralıklı adım)
df.describe().T #CSV sayısal verilere sahip olan sütunların max, min , std… gibi istatiksel değerlerini döndürme.
df.isnull().values.any() #CSV dosyasının içinde herhangi bir verinin eksik olup olmadığını inceleme.
df.isnull().sum() #CSV dosyasının sutunlarında kaç değer olduğunu bulma.
df["sex"].head() #Cinsiyet sutunun nesne tiplerini okuma.
df["sex"].value_counts() #Kadın ve Erkek Cinsiyetlerinin sutunlardaki değişken adetleri

################################################
#Pandas'ta Seçim İşlemleri (Selection in Pandas)
################################################
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0,axis=0).head()

delete_indexes = [1,3,5,7]
df.drop(delete_indexes, axis=0).head(10) #Satır Silme işlemi ayrık veri bulma işlemine yardımcı

#df = df.drop(delete-indexes,axis=0)
#df.drop(delete_indexes,axis=0,inplace=True)

###########################
#Değişkeni İndex'e Çevirmek
###########################
df["age"].head()
df.age.head()

#VeriTabanlarında bazı bilgileri çekmek için indexte kullanacağız sayısal değerler için. Veyahut saat ve dakika için verilebilecek olsa idi.
df.index = df["age"] #Age yani yaş sutununu silmek değil komple sildim
df.drop("age",axis=1).head()

df.drop("age",axis=1,inplace=True)
df.head()

###########################
#Index'i Değişkene Çevirmek.
###########################
df.index
df["age"] = df.index
df.head()
df.drop("age",axis=1,inplace=True)

df.reset_index().head() #Sildiğim satırları geri yükleme
df = df.reset_index()
df.head()

##############################
#Değişkenler Üzerinde İşlemler
##############################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

df["age"].head()
type(df[["age"]].head())

df[["age","alive"]] #İkili sutun gösterme

col_names = ["age","adult_male","alive"] #Görünümde üçlü sutun gösterme.
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop(col_names, axis=1).head()
df.loc[:~df.columns.str.contains("age")].head()

############
#iloc & loc
############
#loc etiket tabanlıdır; bu, satır ve sütun etiketlerine göre satırları ve sütunları belirtmeniz gerektiği anlamına gelir. Alışa gelmiş (0,0) ,-e kadar
#iloc tamsayı konum tabanlıdır, bu nedenle satırları ve sütunları tamsayı konum değerleriyle (0 tabanlı tamsayı konumu) belirtmeniz gerekir. ,-e dahil

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

#iloc: Integer based selection
df.iloc[0:3] #CSV dosyasına göre index sutun olarakta survived göre sıralama
df.iloc[0,0] 

#loc: Label based selection
df.loc[0:3]

df.iloc[0:3,0:3] #Satırdan 3 X Sutundan 3
df.loc[0:3,"age"]

col_names = ["age","embarked","alive"]
df.loc[0:3,col_names]

col_names = ["age","embarked","alive"]
df.loc[0:3,col_names] #FANCY yapısı mulakatlarda da onem arz eder.

#####################################
#Koşullu Seçim(Conditional Selection) , her zaman TRUE FALSE ayıklaması yapar. Koşulun içerisinde True var ise boolean devreye girip koşulu sağlamaya bakacaktır.
#####################################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50] ["age"].count() #SQL SORGUSU gibi gibi

df.loc[df["age"] > 50, ["age","class"]].head()
df.loc[(df["age"] > 50) & (df["sex"] == "male"),["age","class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] =="male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] =="Southampton")),
                ["age","class","embark_town"]]

df_new["embark_town"].value_counts()

##################################################
#Toplulaştırma Ve Gruplama(Aggregation & Grouping)
##################################################
#count(),first(),last(),mean(),median()
#min(),max(),std(),var(),sum(),pivot table

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age":"mean"})
df.groupby("sex").agg({"age":["mean","sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],"survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],"survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],"survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})


#############
# Pivot Table
#############
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


###############################
# Apply ve Lambda Fonksiyonları
###############################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, ["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

###########################
#Birleştirme JOIN İşlemleri
###########################
import numpy as np
import pandas as pd
m = np.random.randint(1,30,size=(5,3))
df1=pd.DataFrame(m,columns=["var1","var2","var3"])
df2 = df1 + 99

pd.concat([df1,df2])
pd.concat([df1,df2],ignore_index=True) #Dizileri birleştirilmesini mevcut dizinleri yok saymak istiyor isek bir nevi indeks tekrarlamaların önüne geçmek istiyorsak.
pd.concat([df1,df2],ignore_index=False)

################################
#Merge İle Birleştirme İşlemleri
################################
df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1,df2) #Veri Setine göre değişken sıralarına göre birleştirme.
pd.merge(df1,df2, on="employees")

#AMAÇ!:Her çalışanın müdürünün bilgisine erişmek isteniliyor.
df3 = pd.merge(df1,df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3,df4)
