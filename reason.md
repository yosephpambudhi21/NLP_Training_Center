Laporan Proyek Machine Learning - Lyonard Gemilang Putra Merdeka Gustiansyah

Domain Proyek
Jakarta merupakan kota yang dapat dikatakan memiliki kualitas udara terburuk di Indonesia. Dilansir dari IQAir, tidak jarang index AQI Jakarta berada di kategori tidak sehat untuk sensitif. Penyebab dari buruknya kualitas di Jakarta adalah banyaknya kendaraan dan pabrik yang membuat polusi udara semakin menumpuk. Akibatnya banyak seali masyarakat yang terkena penyakit pernapasan. Menurut laporan tahun 2021 yang diterbitkan oleh Organisasi Kesehatan Dunia (WHO), 13 orang di seluruh dunia meninggal setiap menit akibat polusi udara dan penyakit serius seperti penyakit kardiovaskular, stroke, dan kanker paru-paru (Perdana & Muklason, 2023). Pada tahun 2002, sebuah studi dari Asian Development Bank memperkirakan bahwa polusi udara berdampak pada lebih dari 90 juta kasus gejala pernapasan dengan estimasi kerugian ekonomi sekitar 1,8 triliun Rupiah. Dari berbagai jenis polutan yang terdapat di udara ambien, ada dua polutan utama yang memiliki dampak merugikan paling besar pada kesehatan manusia, yaitu ozon permukaan (O3) dan PM2.5 (partikulat berdiameter kurang dari 2.5 mikrometer) (Syuhada, 2022). Tidak hanya manusia yang terkena dampaknya, lingkungan sekitar pun menjadi terpengaruh akibat buruknya kualitas udara di Jakarta.

Alasan
Masalah ini harus segera diatasi agar mengurangi atau bahkan dapat mencegah terjadinya penyakit pernafasan yang diakibatkan oleh polusi udara. Dengan diatasi masalah ini akan banyak masyarakat yang sehat bahkan negara pun tidak akan mengalami kerugian ekonomi. Salah satu cara yang dapat diterapkan untuk mengatasi masalah ini adalah dengan membuat model Machine Learning yang dapat mendeteksi kualitas udara yang memberikan peringatan kepada masyarakat sekitar untuk memakai masker agar mengurangi risiko terkena penyakit pernafasan.

Referensi:
Perdana, D., & Muklason, A. (2023). Machine Learning untuk Peramalan Kualitas Indeks Standar Pencemar Udara DKI Jakarta dengan Metode Hibrid ARIMAX-LSTM. ILKOMNIKA: Journal of Computer Science and Applied Informatics, 5(3), 209–222. https://doi.org/10.28926/ilkomnika.v5i3.588
Syuhada, G. (2022, January 19). Dampak Polusi Udara bagi Kesehatan Warga Jakarta. https://rendahemisi.jakarta.go.id/article/174/dampak-polusi-udara-bagi-kesehatan-warga-jakarta#:~:text=Tingkat%20kerusakan%20atau%20keparahan%20dari,5%20mikrometer

Business Understanding
Problem Statements
Bagaimana membangun model machine learning untuk memprediksi kualitas udara berdasarkan parameter pencemar udara?
Algoritma apa yang paling efektif untuk memodelkan prediksi kualitas udara dengan akurasi tinggi?
Goals
Menghasilkan model prediktif yang mampu mengklasifikasikan kualitas udara dari data polutan yang tersedia.
Mengevaluasi dan membandingkan performa dua model machine learning dalam mengklasifikasi kualitas udara.
Solution statements
Menggunakan dua algoritma, yaitu Random Forest Classifier dan Logistic Regression, lalu mengevaluasi kedua model dengan membandingkan akurasi dan F1-score
Melakukan hyperparameter tuning pada kedua model untuk mengoptimalisasi performa.
Memilih model terbaik berdasarkan hasil evaluasi metrik akurasi dan f1-score.
Data Understanding
Data yang digunakan dalam proyek ini adalah "Air Quality Index in Jakarta" yang berisi Indeks Standar Pencemaran Udara yang diukur dari 5 stasiun pemantauan kualitas udara di Jakarta dari Januari 2010 sampai Februari 2025. Dataset yang digunakan adalah ispu_dki_all yang mengandung data AQI gabungan dari keseluruhan stasiun dari Januari 2010 sampai Februari 2025.

Dataset
Dataset: Air Quality Index in Jakarta Dataset

Jumlah data: 5538 baris dan 11 kolom (file: ispu_dki_all.csv)
Terdapat missing value di kolom pm25, so2, dan no2
Terdapat outlier pada hampir semua kolom numerik (PM2.5, PM10, CO, SO2, O3, NO2, max)
Tidak ditemukan duplikat data berdasarkan hasil pemeriksaan df.duplicated().sum()
Variabel-variabel pada Air Quality Index in Jakarta dataset adalah sebagai berikut:
tanggal : Tanggal pencatatan kualitas udara
stasiun : Lokasi stasiun pemantauan
pm25 : Konsentrasi materi partikulat dengan diameter 2,5 mikrometer atau kurang (PM2.5), diukur dalam mikrogram per meter kubik (µg/m³).
pm10 : Konsentrasi materi partikulat dengan diameter 10 mikrometer atau kurang (PM10), diukur dalam mikrogram per meter kubik (µg/m³).
so2 : Konsentrasi sulfur dioksida (SO2), diukur dalam bagian per juta (ppm).
co : Konsentrasi karbon monoksida, diukur dalam bagian per juta (ppm).
o3 : Konsentrasi ozon (O3), diukur dalam bagian per juta (ppm).
no2 : Konsentrasi nitrogen dioksida (NO2), diukur dalam bagian per juta (ppm).
max : Nilai maksimum yang tercatat di antara polutan untuk tanggal dan stasiun tertentu. Nilai ini mewakili konsentrasi tertinggi di antara PM25, PM10, SO2, CO, O3, dan NO2.
critical: Polutan yang mempunyai konsentrasi tertinggi pada tanggal dan stasiun tersebut.
category : Kategori kualitas udara berdasarkan nilai 'maks' yang menggambarkan tingkat kualitas udara.
Tipe Data Setiap Variabel
tanggal, stasiun, critical, categori: string (object)
pm25, pm10, so2, co, o3, no2, max: numerik (float64)
Eksplorasi Data
Boxplot
Outlier
Gambar di atas merupakan boxplot dari kolom pm25. Data pada dataset ini memiliki banyak outlier. Tidak hanya pada kolom itu saja, kolom-kolom lainnya seperti pm10, so2, co, o3, no2, dan max memiliki outlier juga.
Kualitas udara berdasarkan stasiun
Air Quality by Station
Dari bar chart ini, dapat dilihat kualitas udara berdasarkan stasiun. Stasiun lubang buaya memiliki kualitas udara yang tidak sehat terbanyak di antara stasiun lainnya. Meskipun semua stasiun rata-rata memiliki kualitas udara yang sedang, tidak ada satu kota pun yang memiliki kualitas udara yang sangat baik. Hal ini cukup memprihatinkan.
Jumlah Level Kritis Polutan
Critical Levels Count
Polutan PM2.5 merupakan polutan yang paling sering menyentuh titik kritis. Terdapat 1000 data lebih yang menyatakan bahwa polutan PM2.5 mencapai titik kritis.
Korelasi antar fitur
Correlation between each feature
Dari correlation matrix ini, kolom pm25, pm10, dan max memiliki korelasi kuat terhadap label categori, yang menjadikan bahwa ketiga kolom tersebut sangat mungkin relevan untuk model klasifikasi. Kolom stasiun terhadap pm25/pm10/categori juga cukup memiliki hubungan yang menandakan bahwa stasiun juga memiliki peran terhadap kategori kualitas udara.
Tren Historis
Historical Trend
Dilihat dari line chart ini, PM2.5 merupakan polutan paling fluktuatif dan palin sering melonjak tinggi. Apabila dilihat dari pola Polutan seperti PM2.5 dan PM10, sangat memungkinkan bahwa pada 2025 akan mengalami kenaikan lagi dan mungkin akan mencapai puncak pada pertengahan 2025 seperti pada bulan Oktober 2023.
Data Preparation
Dalam pengerjaan proyek ini diterapkan beberapa teknik data preparation, diantara lain:

Menghapus data missing value (dropna())
Menerapkan metode IQR capping untuk menangani outlier
Drop kolom tanggal, max, dan critical karena tidak relevan untuk model klasifikasi
Menerapkan Label Encoding untuk kolom stasiun dan categori (LabelEncoder)
Standarisasi fitur numerik menggunakan StandardScaler
Split data menggunakan train_test_split dengan proporsi 80% untuk data latih dan 20% untuk data uji.
Alasan Tahapan Data Preparation
Data yang bersih dari data duplikat, outlier, missing value dan sudah distandarisasi dapat membuat model tidak bias dan dapat melakukan generalisasi dengan baik.
Kolom tanggal hanya menunjukkan kapan data diambil sehingga tidak relevan untuk digunakan. Kolom seperti max di drop karena meskipun kolom tersebut memiliki korelasi yang kuat dengan categori, kolom ini hanya memberitahu ulang nilai polutan mana yang memiliki nilai paling tinggi. Dengan adanya kolom max, ditakutkan model hanya melihat max dan mengabaikan kontribusi polutan lain. Kolom critical juga di drop karena kolom tersebut juga hanya berisi data dari nama polutan yang paling tinggi konsentrasinya.
Encoding perlu dilakukan agar fitur kategorikal dapat digunakan oleh algoritma machine learning yang akan dibuat.
Standarisasi dilakukan agar mencegah fitur dengan angka besar mendominasi proses training. Dengan dilakukan tahap ini juga dapat memastikan setiap fitur memberi kontribusi yang setara pada pemodelan.
Splitting data dilakukan agar model dapat dilatih pada sebagian besar data dan diuji pada data yang belum pernah dilihat sebelumnya.
Modeling
Terdapat dua model yang digunakan dalam proyek ini, yaitu Random Forest Classifier dan Logistic Regression.

Random Forest Classifier
Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan (decision tree) selama proses training dan menggabungkan prediksi dari semua pohon untuk menentukan hasil akhir. Model ini bekerja dengan prinsip voting untuk klasifikasi dan rata-rata untuk regresi. Setiap pohon dilatih dengan subset data dan subset fitur yang dipilih secara acak.

Parameter yang Digunakan dalam Proses Development
Parameter yang dipakai adalah default.

n_estimators=100: jumlah pohon yang dibuat
criterion='gini': fungsi untuk mengukur kualitas split
max_depth=None: pohon akan berkembang sampai semua daun sempurna
min_samples_split=2: jumlah minimum sampel untuk membagi internal node
min_samples_leaf=1: jumlah minimum sampel di daun
max_features='sqrt': jumlah fitur yang dipertimbangkan pada tiap split
max_leaf_nodes=None: Tidak ada batasan jumlah daun
min_impurity_decrease=0.0: Minimum pengurangan impuritas untuk membagi node
bootstrap=True: menggunakan sampling bootstrap
oob_score=False: Tidak menghitung out-of-bag score
n_jobs=None: Proses dijalankan dalam satu core
random_state=None: tidak ada seed acak yang disetel
verbose=0: Tidak ada logging
warm_start=False: Tidak mempertahankan model sebelumnya
class_weight=None: Tidak menyesuaikan bobot kelas
ccp_alpha=0.0: Complexity pruning tidak digunakan
max_samples=None: Menggunakan seluruh sampel
Kelebihan
Cocok untuk data tabular dan kompleks
Dapat menangani outlier dan missing dengan lebih stabil
Kekurangan
Interpretasi sulit
Memerlukan komputasi lebih besar saat tuning
Tuning yang Diterapkan
Parameter yang dipilih untuk dituning adalah n_estimators, max_depth, min_samples_split, dan max_features. Keempat parameter ini dipilih karena parameter-parameter ini mengontrol kompleksitas model dan mampu membuat model menggeneralisasi dengan baik. Dilakukan tuning menggunakan RandomizedCV untuk mencari hyperparameter optimal dengan value setiap parameter sebagai berikut:

n_estimators: Meningkatkan jumlah pohon diuji pada [50, 100, 150] untuk mengecek stabilitas prediksi agar tidak overfitting.
max_depth: [None, 5, 10, 15] digunakan untuk membatasi overfitting.
min_samples_split: [2, 5, 10] untuk mengatur kapan node dapat di-split lebih lanjut. Nilai lebih tinggi mengurangi overfitting.
max_features: ['sqrt', 'log2'] menentukan jumlah maksimum fitur yang digunakan untuk mencari split terbaik di setiap node. sqrt cocok untuk dataset dengan banyak fitur dan log2 lebih konservatif yang dapat membantu generalisasi.
Logistic Regression
Logistic Regression adalah algoritma linier yang digunakan untuk klasifikasi. Model ini memprediksi probabilitas suatu kelas berdasarkan kombinasi linier dari fitur input, menggunakan fungsi sigmoid/logistik untuk mengkonversi hasil menjadi probabilitas antara 0 dan 1.

Parameter yang Digunakan dalam Proses Development
Parameter yang dipakai adalah default

penalty='l2': Regularisasi L2
dual=False: Solver primal digunakan
tol=1e-4: Toleransi konvergensi
C=1.0: Parameter regulasi
fit_intercept=True: Menambahkan intercept ke model
intercept_scaling=1: Hanya berlaku jika solver 'liblinear'
class_weight=None: Tidak ada penyesuaian bobot antar kelas
random_state=None: Tidak ada seed acak yang disetel
solver='lbfgs': Solver optimisasi default
max_iter=100: Iterasi maksimum
multi_class='auto': Menyesuaikan otomatis metode klasifikasi
verbose=0: Tidak ada logging
warm_start=False: Tidak mempertahankan hasil training sebelumnya
n_jobs=None: Proses dijalankan dalam satu core
l1_ratio=None: Tidak digunakan kecuali regularisasi elasticnet
Kelebihan
Cepat, efisien, dan interpretatif
Cocok untuk klasifikasi linier sederhana
Kekurangan
Lemah untuk menangkap relasi non-linear
Tuning yang Diterapkan:
Parameter yang dipilih untuk dituning adala tol, class_weight, dan max_iter. Ketiga parameter dipilih karena cukup memengaruhi performa model ini. Dilakukan tuning menggunakan RandomizedCV untuk mencari hyperparameter optimal dengan value setiap parameter sebagai berikut:

tol: Nilai toleransi diuji dengan [0.001, 0.05, 0.2] untuk menyesuaikan sensitivitas konvergensi. Nilai tol yang lebih besar memungkinkan training yang lebih cepat, sedangkan nilai tol yang lebih kecil dapat memberikan hasil yang lebih presisi.
class_weight: [None, 'balanced'] untuk mengatasi distribusi tidak seimbang. Nilai 'balanced' akan menyesuaikan bobot kelas secara otomatis berdasarkan frekuensi kelas.
max_iter: [100, 200, 300] untuk memastikan solver mencapai konvergensi.
Dengan menggunakan RandomizedCV untuk mencari hyperparameter optimal secara acak, didapat hyperparameter terbaik seperti gambar di bawah ini:
Tuning Param

Dari 2 model yang sudah dibuat, model yang dipilih untuk klasifikasi kualitas udara adalah Random Forest. Random Forest menghasilkan akurasi dan F1 Score lebih tinggi dibandingkan Logistic Regression.

Evaluation
Metrik yang digunakan untuk mengevaluasi model, yaitu:

Akurasi
F1 Score
Akurasi
Akurasi mengukur proporsi prediksi yang benar dari keseluruhan jumlah prediksi, baik yang benar positif (True Positive/TP) maupun benar negatif (True Negative/TN). Berikut merupakan rumus akurasi:
A c c u r a c y = T P + T N T P + T N + F P + F N

TP (True Positive): Prediksi benar untuk kelas positif
TN (True Negative): Prediksi benar untuk kelas negatif
FP (False Positive): Prediksi salah, model memprediksi positif yang seharusnya negatif
FN (False Negative): Prediksi salah, model memprediksi negatif yang seharusnya positif
F1-Score
F1 Score adalah rata-rata harmonis dari Precision dan Recall. F1 Score sangat penting untuk kasus dengan distribusi kelas tidak seimbang karena mempertimbangkan baik prediksi positif yang tepat maupun semua prediksi positif yang seharusnya. Berikut merupakan rumus F1-Score:
F 1 = 2 × P r e c i s i o n × R e c a l l P r e c i s i o n + R e c a l l

dengan rumus precision:
P r e c i s i o n = T P T P + F P

dan rumus Recall:
R e c a l l = T P T P + F N

Evaluasi Sebelum Tuning
RF Accuracy: 0.9920634920634921
RF F1 Score: 0.9916796348168897
CM RF
Akurasi model Random Forest pada data test adalah sebesar 0.99 begitu juga dengan F1-score. Dilihat pada gambar metrik di atas, model ini hanya memiliki 2 kesalahan prediksi, yaitu label 1 yang diprediksi sebagai 0 sebanyak 1, dan label 2 yang diprediksi sebagai 1 sebanyak 1. Namun, perlu dicari tahu apakah model ini overfitting atau tidak. Salah satu caranya, yaitu dengan membandingkan akurasi test dengan train seperti di bawah ini:
RF Train Accuracy: 1.0
RF Test Accuracy: 0.9920634920634921
RF Train F1 Score: 1.0
RF Test F1 Score: 0.9916796348168897
Ternyata model tersebut mengalami overfitting karena akurasi train lebih besar dibandingkan akurasi tes begitu juga dengan F1-scorenya.

LR Accuracy: 0.9761904761904762
LR F1 Score: 0.9757963693965064
CM LR
Akurasi model Logistic Regression pada data test adalah sebesar 0.97 begitu juga dengan F1-score. Dilihat pada gambar metrik di atas, model ini memiliki lebih banyak kesalahan prediksi dibandingkan Random Forest, yaitu label 1 yang diprediksi sebagai 0 sebanyak 1 dan diprediksi sebagai 2 sebanyak 3, dan label 2 yang diprediksi sebagai 1 sebanyak 2. Dilakukan cara yang sama untuk memeriksa apakah model ini overfitting atau tidak. Berikut perbandingan akurasi dan F1-Score pada train dan tes pada model Logistic Regression:
LR Train Accuracy: 0.9424603174603174
LR Test Accuracy: 0.9761904761904762
LR Train F1 Score: 0.9356552509581937
LR Test F1 Score: 0.9757963693965064
Model ini tidak mengalami overfitting, bahkan akurasi dan F1-score tes lebih tinggi dibandingkan dengan train.

Sejauh ini, Random Forest masih menjadi model yang unggul dibanding Random Forest.

Evaluasi Setelah Tuning
RF Accuracy: 0.9920634920634921
RF F1 Score: 0.9916796348168897
CM RF AT
Setelah dilakukan tuning, akurasi model Random Forest pada data test tetap sebesar 0.99 begitu juga dengan F1-score. Dilihat pada gambar metrik di atas, sama seperti sebelumnya, model ini hanya memiliki 2 kesalahan prediksi, yaitu label 1 yang diprediksi sebagai 0 sebanyak 1, dan label 2 yang diprediksi sebagai 1 sebanyak 1. Berikut perbandingan akurasi dan F1-Score pada train dan tes pada model Random Forest:
RF Train Accuracy: 0.9890873015873016
RF Test Accuracy: 0.9920634920634921
RF Train F1 Score: 0.987401310877358
RF Test F1 Score: 0.9916796348168897
Model lebih baik dari sebelumnya karena model ini tidak lagi menghasilkan akurasi dan F1-score train yang lebih tinggi dari pada tes. Hal ini menandakan bahwa model Random Forest ini tidak lagi overfitting.

LR Accuracy: 0.9801587301587301
LR F1 Score: 0.9797413623201591
CM LR AT
Setelah dilakukan tuning, akurasi model Logistic Regression mengalami peningkatan sehingga menjadi 0.98 begitu juga dengan F1-score yang menjadi 0.979. Dilihat pada gambar metrik di atas, kesalahan yang dibuat model ini berkurang, yaitu label 1 yang diprediksi sebagai 0 sebanyak 1 dan diprediksi sebagai 2 sebanyak 3, dan label 2 yang diprediksi sebagai 1 sebanyak 1. Berikut perbandingan akurasi dan F1-Score pada train dan tes pada model Logistic Regression:
LR Train Accuracy: 0.9424603174603174
LR Test Accuracy: 0.9801587301587301
LR Train F1 Score: 0.9362806614769447
LR Test F1 Score: 0.9797413623201591
Model mengalami peningkatan baik secara akurasi dan F1-Score. Model ini juga tidak overfitting.

Kesimpulannya, dari kedua model yang sudah dibuat, model terbaik yang akan dipilih adalah Random Forest karena model Random Forest memiliki performa yang lebih baik dibandingkan Logistic Regression dan model ini tidak lagi overfitting setelah tuning.
