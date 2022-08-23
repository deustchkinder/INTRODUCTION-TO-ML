#VERİ YAPILARI (DATA STRUCTURES)
#1-Veri Yapılarına Giriş ve Hızlı Bir Özet
#2-Sayılar(Numbers): int, float, complex
#3-Boolean(TRUE-FALSE):bool
#4-Liste(List)
#5-Sözlük(Dictionary)
#6-Demet(Tuple)
#7-Kümeler(Set)

###############################################
# Veri Yapılarına Giriş ve Hızlı Özet
##############################################
#Sayılar : İnteger
x = 46
type(x)

#Sayılar : Float
x = 10.3
type(x)

#Sayılar: Complex
x = 2j + 1
type(x)

#String
x = "Hello Ai Era"
type(x)

#Boolean
True 
False
type(True)

5 == 4
3 == 2
1 == 1
type(3 == 2)

#List
x = ["btc","eth","xrp"]
type(x)

#Dictionary
x = {"name":"Emre", "Age":23}
type(x)

#Set
x = {"python","ml","ds"}
type(x)

#Tuple
x = ("python","ml","ds")
type(x)

#NOT:Liste,Demet,Küme ve Sözlük veri yapılarını aynı zamanda Python Collections(Arrays) olarak geçmektedir.

###############################################
# Sayılar (Numbers): int, float, complex
###############################################
a = 5
b = 10.5

a * 3
a / 7 
a * b / 10
a **2

#######################
# Tipleri değiştirmek
#######################
int(b)
float(a)
int(a * b / 10)

c = a * b / 10
int(c)
float(c)

###############################################
# Karakter Dizileri (Strings)
###############################################
print("Emre")
print('Emre')

"Emre"
name = "Emre"
name = 'Emre'

###############################
# Çok Satırlı Karakter Dizileri
###############################
"""Veri Yapıları: Hızlı Özet,
Sayılar(Numbers): int, float, complex,
Karakter Dizileri(Strings):str,
List,Dictionary,Tuple,Set,Boolean(TRUE-FALSE): bool"""

long_str = """Veri Yapıları: Hızlı Özet,
Sayılar(Numbers): int, float, complex,
Karakter Dizileri(Strings):str,
List,Dictionary,Tuple,Set,Boolean(TRUE-FALSE): bool"""

##############################
#Karakter Dizilerinin Elemanlarına Erişimek
##############################
name
name[0]
name[3]
name[2]
name[6]

######################################
#String İçerisinde Karakter Sorgulamak
######################################
long_str
"veri" in long_str
"Veri" in long_str
"bool" in long_str

##################################
#String(Karakter Dizisi) Metodları
##################################
dir(str)
dir(int)
dir(bool)

########
#LEN()
########
name = "Emre"
type(name)
type(len)

len(name)
len("emredemirkan")
len("miuul!")

############################################
#Upper() & Lower(): Küçük- Büyük Dönüşümleri
############################################
"miuul".upper()
"MİUUL".lower()

#   type(upper) #String index out of range
#   type(upper()) #Name upper is not defined

################################
#Replace(): Karakter değiştirir.
################################
hi = "Hello AI Era"
hi.replace("l","p")

###########################
#Split() : Karakteri Böler.
###########################
"Hello AI Era".split()

#####################
#Strip(): Kırpar
#####################
" ofofo".strip()
"ofofo".strip("o")

#################################
#Capitalize(): İlk harfi büyütür.
#################################
"foo".capitalize()
dir("foo")
"foo".startswith("f")

############
#Liste(List)
############
#-Değiştirilebilir, Sıralıdır, İndex İşlemleri yapılabilir, Kapsayıcıdır.
notes = [1,2,3,4]
type(notes)
names = ["a","b","v","d"]
not_nam = [1,2,3,"a","b",True,[1,2,3]]

not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]

type(not_nam[6])
type(not_nam[6][1])
notes[0] = 99
not_nam[0:4]

##############################
#Liste Metotları(List Methods)
##############################
dir(notes)

#len(): Built in python fonksiyonu(Gömülü Fonksiyonlar), boyut bilgisi.Variable Explorerdaki size
len(notes)
len(not_nam)

#append(): Listeye eleman ekler.
notes
notes.append(100)

#pop():İndex'e göre siler.
notes.pop(0)

#insert():Listeyi indexe ekler.
notes.insert(2,99)
notes

####################
#Sözlük {Dictionary}
####################

#Değiştirilebilir, Sırasızdır,(3.7 sürümünden sonra sıralıdır.),Kapsayıcıdır.
#{KEY:VALUE}
dictionary = {"REG":"Regression",
              "LOG":"Logistic Regression",
              "CART":"Classfication And Regression"}
dictionary["REG"]

dictionary = {"REG":["RMSE",10],
              "LOG":["MSE",20],
              "CART":["SSE",30]}

dictionary = {"REG":10,
              "lOG":20,
              "CART":30}
dictionary["CART"][1]

#KEY(ANAHTAR) sorgulama
"YSA" in dictionary

#Key'e göre value'ya erişmek. KİME GÖRE NEYE GÖRE HOCAM?
dictionary["REG"]
dictionary.get("REG"), #GET FONKSİYON RETURN BİR FONKSİYON OLUP, ALDIĞI KEY DEĞERİNİN KARŞILIĞINI GERİ DÖNDÜREN 

#Value değiştirmek
dictionary["REG"] = ["YSA",10]
dictionary

#Tüm key'lere erişmek
dictionary.keys()

#Tüm value'lere erişmek
dictionary.values()

#Tüm çiftleri tuple halinde listeye çevirme
dictionary.items()

#Key-Value değerini güncellemek
dictionary.update({"REG":11})
dictionary

#Yeni key-value eklemek
dictionary.update({"RF":10})

###############
#Demet (Tuple)
###############
#Değiştirilemez, Sıralıdır, Kapsayıcıdır.

t = ("john","mark",1,2)
type(t)

t[0]
t[0:3]

t[0] = 99

t = list(t)
t[0] = 99
t=tuple(t)
t

##############
#Set(Kümeler)
##############
#Değiştirilebilir, Sırasız + Eşsizdir, Kapsayıcıdır.

#difference(): İki kümenin farkı
set1 = set([1,3,5])
set2 = set([1,2,3])

#set1'de olup set2'de olmayanlar
set1.difference(set2)
set1 -set2

#set2'de olup set1'de olmayanlar
set2.difference(set1)
set2 -set1

#symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#intersection(): İki kümenin kesişimi
set1 = set([1,3,5])
set2 = set([1,2,3])

set1.intersection(set2)
set2.intersection(set1)

set1 & set2

#union():İki kümenin birleşimi
set1.union(set2)
set2.union(set1)

#isdisjoint(): İki kümenin kesişimi boş mu?
set1 = set([7,8,9])
set2 = set([5,6,7,8,9,10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

#issuperset(): Bir küme diğer kümeyi kapsıyor mu?
set2.issuperset(set1)
set1.issuperset(set2)
