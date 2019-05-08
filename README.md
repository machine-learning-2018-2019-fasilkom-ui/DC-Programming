# DC-Programming

#### Our team
- Ibnu Sofian Firdaus (1606893986)
- Ilham Darmawan Candra Purnama (1606882351)
- Luthfi Dzaky Saifuddin (1606889830)
#### Deskripsi Proyek
Sentiment Analysis merupakan salah satu implementasi dari machine learning. Program tersebut akan mencoba mengidentifikasi sentiment dari teks dalam bahasa inggris. Biasanya sentiment hanya terbagi menjadi dua representatif, sentiment positif dan sentiment negatif. Pada proyek kali ini kami akan lebih fokus untuk mendeteksi emosi dari penulis teks tersebut, jadi bukan hanya dbagi menjadi 2 representatif, namun menjadi 13 representatif emosi. Kami mendapatkan data tersebut dari website Kaggle, data tersebut mengandung 30000 baris kata yang sudah diidentifikasi emosinya.

#### Metode
Teknik yang akan digunakan dalam implementasi machine learning pendeteksian emosi ini adalah Supervised Machine Learning dengan model Naive Bayes. Supervised Machine Learning adalah sebuah pendekatan pembelajaran sebuah fungsi yang dapat memetakan sebuah input kedalam sebuah output berdasarkan contoh data (training set) berupa pasangan input-output. Naive Bayes adalah sebuah model machine learning yang merupakan sebuah probabilistic classifier dengan menerapkan prinsip Bayes’ Theorem dengan asumsi kuat (naive) bahwa setiap features atau atribut data bersifat independen satu sama lain. Prinsip Bayes’ Theorem sendiri dapat dituliskan sebagai berikut :
```
p(A | B) = ( p(B | A)  * p(A) ) / p(B)
```

#### How to use

Dapat diakses pada link berikut:

[Klik disini](https://www.google.com)

atau

melakukan perintah berikut setelah melakukan git clone:
```
python –m venv env
env\Scripts\activate.bat 
pip install –r requirements.txt
```

setelah melakukan perintah di atas, lakukan perintah berikut:
```
python manage.py runserver 8000
```

Setelah itu dapat di akses pada
[localhost](localhost:8000) 

#### Referensi

- https://www.kaggle.com/c/sa-emotions/data
- Rana, Shweta & Singh, Archana. 2016. Comparative analysis of sentiment orientation using SVM and Naive Bayes techniques. 2nd International Conference on Next Generation Computing Technologies (NGCT). India.