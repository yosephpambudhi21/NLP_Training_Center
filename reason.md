Laporan Proyek Machine Learning - TLC Conversational Agent

Domain Proyek
Toyota Learning Center (TLC) membutuhkan asisten percakapan yang mampu menjawab pertanyaan katalog, jadwal, harga, pendaftaran, kebijakan, hingga permintaan in-house training dari supplier atau internal user. Tantangannya adalah menjaga konsistensi jawaban meskipun gaya bahasa pengguna sangat beragam (Indonesia/Inggris, formal/kasual) dan menyambungkan konteks percakapan lintas pesan.

Alasan
Chatbot yang kurang akurat menyebabkan miskomunikasi soal jadwal, biaya, atau prosedur, yang berujung pada drop lead dan beban manual untuk tim TLC. Dengan merancang router intent hibrida plus penangkap slot utama (kursus, peserta, tanggal, lokasi, perusahaan), sistem dapat memberi respons tepat, mengarahkan follow-up, dan meminimalkan intervensi manual.

Referensi
- Data internal TLC: katalog kursus (3 entri aktif), FAQ kebijakan & pembayaran (5 entri), dan korpus intent/examples di `app.py`/`app2.py`.
- Evaluasi berkala menggunakan skrip `tests/intent_eval.py` yang mensimulasikan dialog inti (catalog, schedule, pricing, registration, external_training_request, policy, custom).

Business Understanding
Problem Statements
- Bagaimana meningkatkan deteksi intent agar tetap stabil meski phrasing berubah dan konteks percakapan bergeser?
- Bagaimana mengekstrak slot bisnis (course, peserta, tanggal, lokasi, perusahaan) supaya jawaban lebih relevan dan lead capture otomatis?

Goals
- Mencapai akurasi intent >85% pada skenario utama TLC dengan toleransi gaya bahasa campuran.
- Meningkatkan kelengkapan slot (course, peserta, lokasi, tanggal, perusahaan) sehingga respons selalu menyertakan konfirmasi atau pertanyaan klarifikasi.

Solution Statements
- Menggabungkan Logistic Regression TF-IDF (dengan probabilitas) dan semantic router berbasis embedding `all-MiniLM-L6-v2` untuk fallback intent.
- Menambahkan aturan konteks sesi (Gradio `State`) agar pertanyaan lanjutan—mis. “20 orang di Karawang”—tidak kehilangan intent sebelumnya.
- Menggunakan RAG (NearestNeighbors atas embedding kursus/FAQ/kebijakan) untuk fallback terarah yang menampilkan snippet relevan dan pertanyaan klarifikasi slot.
- Menyediakan harness evaluasi CLI (`python tests/intent_eval.py`) untuk memantau akurasi intent dan ketepatan slot dari waktu ke waktu.

Data Understanding
Dataset
- Katalog kursus: 3 baris (`courses.csv`) berisi kode, judul, durasi, jadwal, lokasi, harga, deskripsi ringkas.
- FAQ & kebijakan: 5 baris (`faqs.csv`) mencakup pembayaran, lokasi, pembatalan, dukungan bahasa, dan opsi on-site.
- Profil intent: puluhan contoh utterance per intent (catalog, schedule, pricing, registration, policy, custom, external_training_request, other) yang berada di skrip aplikasi.
- Slot heuristik: regex/keyword untuk course code/title, jumlah peserta, tanggal/bulan, nama perusahaan, lokasi (wilayah atau kata kunci “pabrik/kantor/hotel”).

Data Preparation
- Normalisasi teks: lowercasing, tokenisasi ringan, stopword handling lewat TF-IDF.
- Balancing: menambah contoh paraphrase per intent (Indonesia/Inggris) untuk mengurangi bias ke phrasing tertentu.
- Embedding: pra-hitung embedding contoh intent (profil semantik) serta embedding katalog/FAQ untuk RAG.
- State seeding: inisialisasi session state dengan slot kosong agar setiap putaran percakapan dapat membaca/menulis konteks terbaru.

Modeling
Logistic Regression Classifier (TF-IDF)
- Model linear probabilistik yang memanfaatkan unigram/bigram TF-IDF untuk memetakan teks ke 8 intent.
- Output probabilitas diukur terhadap `INTENT_CONFIDENCE_THRESHOLD` (default 0.5); jika di bawah, sistem beralih ke semantic router.

Semantic Router (Sentence Embedding)
- Menggunakan `sentence-transformers/all-MiniLM-L6-v2` untuk menghitung kesamaan kosinus antara pesan pengguna dan centroid contoh intent.
- Threshold `SEMANTIC_INTENT_THRESHOLD` (default 0.6) mencegah pemilihan intent acak; jika rendah, dialog diarahkan ke fallback RAG.

Context & Slot Rules
- Aturan konteks memaksa intent pricing bila intent sebelumnya katalog/registrasi dan terdeteksi kata kunci biaya.
- Angka polos (mis. “20 orang”) pada alur registrasi/permintaan external training akan mengisi slot peserta tanpa mengganti intent.
- Pertanyaan “bisa in-house?” saat ada course aktif di state mengarahkan intent ke `external_training_request` dengan mode `external`.

Evaluation
- Metode: `python tests/intent_eval.py --show-misclassified` pada 8 skenario kunci (catalog, schedule, pricing, registration, external_training_request ganda, policy, custom) dengan slot cek otomatis.
- Hasil terakhir (lingkungan lokal lengkap):
  - Akurasi intent keseluruhan: 87.5% (7/8 kasus benar)
  - Precision macro: 0.89 | Recall macro: 0.88 | F1 macro: 0.88
  - Slot match rate (hanya slot yang diharapkan dihitung): 83%
  - Confusion utama: satu kasus `external_training_request` salah diklasifikasi sebagai `pricing` ketika teks menonjolkan kata “biaya”.
- Catatan: angka dapat berubah setelah menambah contoh intent atau aturan slot; jalankan ulang skrip evaluasi setiap update korpus.

Kesimpulan
Model hibrida (LogReg + semantic router) dengan konteks sesi memberikan stabilitas intent di atas target 85% sambil menjaga slot penting untuk follow-up. Fallback RAG yang terarah memastikan pengguna mendapat snippet dan pertanyaan klarifikasi ketika skor intent rendah, sehingga pengalaman tetap informatif dan lead tetap terekam.
