##########################################
#VERİ GÖRSELLEŞTİRME: MATPLOTLİB & SEABORN
##########################################
#Python dünyasıda veri görselleştirme çokta kendimizi ifade etmeye yaramaz. PowerPI kullanmalıyız.

#MATPLOTLİB: Kategorik Değişken:Sutun Grafik, Count Plot Bar & Sayısal Değişken: Histogram, BoxPlot

#Kategorik Değişken Görselleştirme
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts().plot(kind='bar')
plt.show()

#Sayısal Değişken Görselleştirme
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"]) #Baskılama Yöntemi
plt.show()

#Matplotlib'in Özellikleri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

#####
#PLOT
#####
x = np.array([1,8])
y = np.array([0,150])
plt.plot(x,y)
plt.show()

x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])
plt.plot(x,y)
plt.show()

plt.plot(x,y,'o') #Ordinat noktasına göre belirgeç.
plt.show()

#######
#MARKER
#######
y= np.array([13,28,11,100])
plt.plot(y,marker='o')
plt.show()

plt.plot(y,marker='*')
plt.show()

markers = ['o','*','.',',','x','X','+','P','s','D','d','p','H','h']

#####
#LINE
#####
y=np.array([13,28,11,100])
plt.plot(y,linestyle="dashdot",color="r")
plt.show()

###############
#MULTIPLE LINES
###############
x = np.array([23,18,31,10])
y = np.array([13,28,11,100])
plt.plot(x)
plt.plot(y)
plt.show()

#######
#LABELS
#######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x = np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])

plt.plot(x,y)
plt.title("Bu ana başlık!") #Başlık Oluşturma

plt.xlabel("X ekseni isimlendirmesi")
plt.ylabel("Y ekseni isimlendirmesi")

plt.grid()
plt.show()

#########
#SUBPLOTS
#########

#plot 1
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)

#plot 2
x = np.array([8,8,9,9,10,15,11,15,12,15])
y = np.array([24,20,26,27,28,29,30,30,30,30])
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)
plt.show()

#ORNEK 3 grafiği, 1satır X 3sutun olarak konumlamak.
#plot 1
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,3,1)
plt.title("1")
plt.plot(x,y)

#plot 2
x = np.array([8,8,9,9,10,15,11,15,12,15])
y = np.array([24,20,26,27,280,29,30,30,30,30])
plt.subplot(1,3,2)
plt.title("2")
plt.plot(x,y)

#plot 3
x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1,3,3)
plt.title("3")
plt.plot(x,y)

plt.show()

#Seaborn: Sayısal Değişken Görselleştirme, Gelişmiş fonksiyonel keşifçi veri analizi...
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar')
plt.show()

################################
#Sayısal Değişken Görselleştirme
################################
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

##########################################
#GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ
#(ADVANCED FUNCTIONAL EDA)
#1-Genel Resim
#2-Kategorik Değişken Analizi(Analysis of Categorical Variables)
#3-Sayısal Değişken Analizi(Analysis of Numerical Variables)
#4-Hedef Değişken Analizi(Analysis of Target Variable)
#5-Korelasyon Analizi(Analysis of Correlation)

##############
#1.GENEL RESİM
##############
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")

df.head()
df.tail()
df.shape()
df.info()
df.columns
df.index
df.describe().T 
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("########################SHAPE#########################")
    print(dataframe.shape)
    print("########################TYPES#########################")
    print(dataframe.dtypes)
    print("#######################HEAD###########################")
    print(dataframe.head(head))
    print("#######################TAİL###########################")
    print(dataframe.tail(head))
    print("#######################NA#############################")
    print(dataframe.isnull().sum())
    print("#######################QUANTİLES######################")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

df = sns.load_dataset("flights")
check_df(df)

#################################################################
#2.KATEGORİK DEĞİŞKEN ANALİZİ (ANALYSİS OF CATEGORİCAL VARİABLES)
#################################################################
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]

cat_bur_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
100*df["survived"].value_counts() / len(df)

def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################")

        if plot : 
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)
 
def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
        
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#############################################")
        
        if plot:
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                           "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
        
        print("##############################################")
        
        if plot:
            sns.countplot(x=dataframe[col_name],data=dataframe)
            plt.show(block=True)
            
cat_summary(df,"adult_male", plot=True)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###############################################")
    
cat_summary(df,"sex")

#############################################################
#3.SAYISAL DEĞİŞKEN ANALİZİ (ANALYSİS OF NUMERİCAL VARİABLES)
#############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]



df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in cat_cols]
#Aykırı değer tespiti yapmasındaki ilk gözlem olarak nitelendirebilirim.
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)
    
##########################################################
#DEĞİŞKENLERİN YAKALANMASI VE İŞLEMLERİN GENELLEŞTİRİLMESİ
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()
df.info()

#DOCSTRİNG
def grab_col_names(dataframe,cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    
    Parameters
    ----------
    dataframe: dataframe
        degisken isimleri alınmak istenen dataframe'dir.
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int,float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri.
        
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_bur_car: list
        Kategorik görünümlü kardinal değişken listesi
        
    Notes
    -----
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.
    """

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    cat_th : TYPE, optional
        DESCRIPTION. The default is 10.
    car_th : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.
    
    cat_cols  = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category","object","bool"]]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

# BONUS
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)

######################################################
#4-HEDEF DEĞİŞKEN ANALİZİ(ANALYSİS OF TARGET VARİABLE)
######################################################
import numpy as np
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

####################################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
####################################################
df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

##################################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
##################################################
df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived","age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)
    
##############################################
#5-KORELASYON ANALİZİ(ANALYSİS OF CORRELATION)
##############################################
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
df = pd.read_csv("C:/Users/Monster/Desktop/pythonProgramlama (1)/pythonProgramlama/python_for_data_science/data_analysis_with_python/datasets/breast_cancer.csv")
df = df.iloc[:,1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int,float]]
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

############################################
#Yüksek Korelasyonlu Değişkenlerin Silinmesi
############################################
cor_matrix = df.corr().abs()
cor_matrix

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000

#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any (upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]
df.drop(drop_list,axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr,cmap="RdBu")
        plt.show()
    return drop_list 

high_correlated_cols(df)
drop_list = high_correlated_cols(df,plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets\fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)
len(df.drop(drop_list, axis=1).columns)
type(adsa)

