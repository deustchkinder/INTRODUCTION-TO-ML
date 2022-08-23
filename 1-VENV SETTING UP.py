#########################################################################
#Virtual Enviorement (Sanal Ortam) ve Package Management (Paket Yonetimi) 
#########################################################################

#Yazacagimiz komutlar bir Terminal urunudur,lutfen console'da yazmayiniz!

#Sanal ortamlarin listelenmesi: 
    conda env list
#Sanal ortam olusturulmasi: 
    conda create -n myenv
#Sanal ortami aktif etme: 
    conda activate myenv
#Yukleme paketlerin listelenmesi: 
    conda list
#Paket yukleme : 
    conda install numpy
#Ayni anda birden fazla paket yukleme :
    conda install numpy scipy pandas
#Paket Silme:
    conda remove package_name
#Belirli bir versiyona gore paket yukleme:
    conda install numpy = 1.20.1
#Paket yukseltme:
    conda upgrade numpy
#Tum paketlerin yukseltilmesi:
    conda upgrade -all
#pip: PYPI(Python Package Index) paket yonetim araci
#Paket Yukleme:
    pip install pandas
#Paket Yukleme:
    pip install pandas
#Paket yukleme versiyonuna gore: 
    pip install pandas == 1.4.1


