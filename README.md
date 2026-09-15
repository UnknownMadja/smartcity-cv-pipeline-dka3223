# DKA3223 – AI UNTUK COMPUTER VISION

## 1.0 PENGENALAN PROJEK

Projek ini dibangunkan bagi kursus **DKA3223 – AI Untuk Computer Vision** dengan menggunakan persekitaran **Google Colab** dan teknologi **Computer Vision**.

Projek ini memfokuskan kepada dua komponen utama iaitu:

* **Convolutional Neural Network (CNN)** untuk klasifikasi imej.
* **YOLO11n** untuk pengesanan objek dan real-time inference.

Tujuan utama projek ini adalah untuk membangunkan, menguji dan mendokumentasikan model Computer Vision secara sistematik serta mengamalkan prinsip **Responsible AI** dalam proses pembangunan.

Semua fail projek disusun dalam repositori **GitHub** bagi memudahkan pengurusan kod, version control dan rujukan pada masa akan datang.

---

## 2.0 OBJEKTIF PROJEK

Objektif projek ini adalah:

1. Membangunkan dan menguji model Computer Vision menggunakan Google Colab.
2. Melaksanakan simulasi **Convolutional Neural Network (CNN)** untuk klasifikasi imej.
3. Melaksanakan simulasi **YOLO11n** bagi pengesanan objek.
4. Membuat pembetulan terhadap ralat kod dan memastikan notebook dapat dijalankan dengan baik.
5. Menguji kesan perubahan **confidence threshold** terhadap jumlah objek yang dikesan.
6. Melaksanakan inferens terhadap video dengan jumlah frame yang ditetapkan.
7. Mengamalkan pengurusan repositori dan version control yang sistematik.
8. Mengambil kira aspek etika, privasi data, hak cipta dan tanggungjawab pengguna dalam pembangunan AI.

---

## 3.0 PERSEKITARAN PEMBANGUNAN

Projek ini dibangunkan menggunakan teknologi dan persekitaran berikut:

| Teknologi / Platform | Kegunaan                                  |
| -------------------- | ----------------------------------------- |
| Google Colab         | Persekitaran pembangunan dan pengujian    |
| Python               | Bahasa pengaturcaraan                     |
| PyTorch              | Pembangunan dan latihan model CNN         |
| YOLO11n              | Pengesanan objek                          |
| GitHub               | Pengurusan repositori dan version control |
| NVIDIA Tesla T4 GPU  | Pemprosesan dan inferens YOLO11n          |

Semakan persekitaran sistem menunjukkan bahawa **CUDA GPU tersedia** dan peranti YOLO ditetapkan kepada GPU untuk membantu mempercepatkan proses inferens.

---

## 4.0 STRUKTUR PROJEK

Struktur repositori projek disusun secara sistematik bagi memudahkan pengurusan fail, kod dan dokumentasi.

```text
DKA3223-Computer-Vision/
│
├── README.md
│
├── notebooks/
│   ├── CNN_Simulation.ipynb
│   └── YOLO11n_Object_Detection.ipynb
│
├── results/
│   └── detection_results/
│
└── documentation/
    └── project_report.pdf
```

Nama fail dan folder digunakan secara jelas supaya kandungan projek mudah dikenal pasti, diselenggara dan diuruskan.

---

# 5.0 SIMULASI CONVOLUTIONAL NEURAL NETWORK (CNN)

Komponen pertama projek ialah simulasi **Convolutional Neural Network (CNN)**.

Notebook CNN telah diperiksa dan beberapa pembetulan dibuat supaya program dapat dijalankan dengan lebih baik serta memenuhi spesifikasi tugasan.

## 5.1 Dataset Path

Pembetulan dilakukan pada laluan dataset supaya lebih fleksibel.

Folder `train` dan `test` dibina menggunakan `os.path.join()` dan kewujudan folder disemak sebelum dataset dimuatkan.

Contoh:

```python
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
```

---

## 5.2 GPU dan Reproducibility

Semakan GPU ditambah bagi memastikan peranti yang digunakan dapat dikenal pasti.

`torch.cuda.manual_seed_all(42)` digunakan bagi membantu proses **reproducibility** apabila latihan menggunakan CUDA.

Nama GPU aktif juga dipaparkan menggunakan:

```python
torch.cuda.get_device_name(0)
```

---

## 5.3 Visualisasi Dataset

Proses **denormalization** digunakan sebelum imej dipaparkan bagi mendapatkan paparan imej yang lebih sesuai.

Selain itu, fungsi berikut digunakan:

```python
torch.clamp(image, 0, 1)
```

Fungsi tersebut memastikan nilai piksel berada dalam julat yang sesuai sebelum imej dipaparkan.

---

## 5.4 Saiz Input Model

Masalah ketidakpadanan saiz **feature map** pada bahagian `fully connected layer` diperbaiki.

`AdaptiveAvgPool2d((1,1))` digunakan bagi menghasilkan feature map yang lebih sesuai dan mengurangkan masalah ketidakpadanan dimensi input.

Contoh:

```python
nn.AdaptiveAvgPool2d((1, 1))
```

---

## 5.5 Bilangan Kelas

Bilangan kelas tidak lagi ditetapkan secara manual.

Sebaliknya, jumlah kelas diambil secara terus daripada dataset:

```python
num_classes = len(train_dataset.classes)
```

Kaedah ini menjadikan model lebih fleksibel sekiranya bilangan kelas dalam dataset berubah.

---

## 5.6 Confidence Score

**Softmax** digunakan bagi mendapatkan kebarangkalian bagi setiap kelas.

Nilai kebarangkalian tertinggi digunakan sebagai **confidence score** dan dipaparkan dalam bentuk peratus.

Contoh:

```python
probabilities = torch.softmax(outputs, dim=1)
confidence, predicted = torch.max(probabilities, 1)
```

---

# 6.0 SIMULASI OBJECT DETECTION MENGGUNAKAN YOLO11n

Komponen kedua projek ialah simulasi **Object Detection menggunakan YOLO11n**.

Model `yolo11n.pt` digunakan untuk mengesan objek pada imej dan video.

Lima pembetulan utama telah dilakukan pada kod seperti berikut.

---

## 6.1 FIX ME 1 – CONF_THRESHOLD

Nilai confidence threshold ditetapkan kepada **0.25**.

```python
CONF_THRESHOLD = 0.25
```

Nilai ini digunakan untuk menentukan tahap keyakinan minimum sesuatu pengesanan sebelum ia diterima sebagai objek yang dikesan.

---

## 6.2 FIX ME 2 – IOU_THRESHOLD

Nilai IoU threshold ditetapkan kepada **0.45**.

```python
IOU_THRESHOLD = 0.45
```

Nilai ini digunakan dalam proses **Non-Maximum Suppression (NMS)** bagi mengurangkan pengesanan bertindih terhadap objek yang sama.

---

## 6.3 FIX ME 3 – CONF_VALUES

Empat nilai confidence digunakan bagi menjalankan eksperimen.

```python
CONF_VALUES = [0.25, 0.45, 0.65, 0.85]
```

Eksperimen ini digunakan untuk membandingkan kesan perubahan confidence threshold terhadap jumlah objek yang dikesan.

---

## 6.4 FIX ME 4 – OBJECT_CLASS_FIELD

Kunci `"nama_kelas"` digunakan bagi mendapatkan nama kelas objek daripada rekod pengesanan.

```python
OBJECT_CLASS_FIELD = "nama_kelas"
```

Maklumat ini membolehkan pengiraan objek berdasarkan kelas dilakukan.

---

## 6.5 FIX ME 5 – MAX_FRAMES

Jumlah frame maksimum ditetapkan kepada **120 frame**.

```python
MAX_FRAMES = 120
```

Tetapan ini digunakan bagi mengawal tempoh pemprosesan video serta penggunaan sumber GPU dan RAM.

---

# 7.0 KEPUTUSAN PENGESANAN OBJEK

Berdasarkan output ujian, model **YOLO11n** berjaya mengesan beberapa objek seperti `person` dan `bus`.

Dalam salah satu output terperinci, sebanyak **5 objek** telah dikesan.

| Objek  | Confidence |
| ------ | ---------: |
| Bus    |     94.02% |
| Person |     88.82% |
| Person |     87.83% |
| Person |     85.58% |
| Person |     62.19% |

### Jumlah Objek Mengikut Kelas

```text
person : 4
bus    : 1
Jumlah : 5
```

Keputusan ini menunjukkan bahawa model dapat mengenal pasti beberapa objek dalam imej ujian bersama nilai confidence bagi setiap pengesanan.

---

# 8.0 EKSPERIMEN CONFIDENCE THRESHOLD

Eksperimen dijalankan menggunakan beberapa nilai confidence threshold bagi melihat perubahan jumlah objek yang dikesan.

| Confidence Threshold | Jumlah Objek | Analisis                                                    |
| -------------------: | -----------: | ----------------------------------------------------------- |
|                 0.25 |            5 | Mengesan objek utama dan objek separa terlindung.           |
|                 0.45 |            5 | Masih mengesan objek dengan tahap keyakinan yang mencukupi. |
|                 0.65 |            4 | Sebahagian objek dengan keyakinan lebih rendah ditapis.     |
|                 0.85 |            4 | Hanya objek dengan keyakinan tinggi diluluskan.             |

### Analisis

Eksperimen menunjukkan bahawa peningkatan **confidence threshold** boleh menyebabkan pengesanan dengan nilai keyakinan lebih rendah ditapis.

Secara umum:

```text
Confidence Threshold Rendah
        ↓
Lebih banyak pengesanan diterima

Confidence Threshold Tinggi
        ↓
Pengesanan dengan confidence rendah ditapis
```

---

# 9.0 REAL-TIME / VIDEO INFERENCE

Model **YOLO11n** turut digunakan untuk pemprosesan video.

Sebanyak **120 frame** telah diproses dan video output disimpan sebagai:

```text
kv_tron_detection_output.mp4
```

Jumlah detection yang direkodkan sepanjang proses video adalah:

| Kelas Objek | Jumlah Detection |
| ----------- | ---------------: |
| Person      |              597 |
| Car         |              120 |
| Truck       |              120 |
| Dog         |               21 |

Proses video berjaya diselesaikan dan semua **lima FIX ME** telah disahkan selesai tanpa ralat.

---

# 10.0 SUMBER KOD DAN DATASET

Sebarang kod rujukan atau sumber dataset yang digunakan dalam projek hendaklah dinyatakan dengan jelas dalam dokumentasi projek.

Penggunaan kod, dataset, model atau bahan daripada pihak lain perlu menghormati **hak cipta** dan **lesen** yang berkaitan.

Kod dan dataset tidak boleh dianggap sebagai hasil milik sendiri sekiranya ia diperoleh daripada sumber luar.

Atribusi yang sesuai perlu diberikan kepada pemilik atau sumber asal.

> **Nota:** Maklumat URL atau nama sumber asal hendaklah ditambah berdasarkan sumber sebenar kod dan dataset yang digunakan dalam notebook.

---

# 11.0 ETIKA DAN RESPONSIBLE AI

Pembangunan projek ini mengambil kira prinsip **Responsible AI** dan etika profesional.

Antara aspek yang diberi perhatian ialah:

* Menghormati hak cipta dan harta intelek.
* Menyatakan sumber asal kod dan dataset.
* Tidak mendakwa kod atau dataset pihak lain sebagai hasil sendiri.
* Menjaga privasi data yang digunakan.
* Mengelakkan penggunaan data yang mengandungi maklumat peribadi tanpa kebenaran.
* Mendokumentasikan limitasi model secara jelas.
* Memastikan pengguna memahami bahawa output AI tidak semestinya sentiasa tepat.
* Menggalakkan semakan manusia sebelum keputusan berdasarkan model digunakan.

Repositori projek ditetapkan sebagai **Private Repository** bagi membantu mengawal akses kepada fail dan bahan projek.

---

# 12.0 PRIVASI DAN KESELAMATAN DATA

Data yang digunakan dalam pembangunan Computer Vision perlu dikendalikan secara bertanggungjawab.

Sekiranya dataset mengandungi imej manusia atau maklumat yang boleh mengenal pasti individu, privasi perlu diberi keutamaan.

Fail yang mengandungi maklumat sensitif seperti:

```text
credentials
password
API key
maklumat peribadi
```

tidak sepatutnya dimasukkan ke dalam repositori GitHub.

Repositori **Private** digunakan bagi mengehadkan akses kepada bahan projek dan mengurangkan risiko pendedahan fail yang tidak sepatutnya dikongsi secara umum.

---

# 13.0 LIMITASI MODEL

Model Computer Vision mempunyai beberapa limitasi. Prestasi model bergantung kepada dataset, keadaan imej dan konfigurasi model yang digunakan.

Antara faktor yang boleh mempengaruhi keputusan ialah:

* Kualiti dan kepelbagaian dataset.
* Pencahayaan imej.
* Saiz dan kedudukan objek.
* Objek yang terlindung atau sebahagiannya terlindung.
* Perubahan persekitaran daripada data latihan.
* Nilai confidence threshold yang digunakan.

**Confidence score tidak bermaksud bahawa sesuatu ramalan adalah 100% benar.**

Ia hanya menunjukkan tahap keyakinan model terhadap pengesanan atau klasifikasi tersebut.

---

# 14.0 TANGGUNGJAWAB PENGGUNA

Pengguna bertanggungjawab menggunakan model AI secara beretika dan berhati-hati.

Hasil daripada model perlu dianggap sebagai bantuan kepada pengguna dan bukan semestinya keputusan akhir.

Pengguna perlu:

1. Menyemak keputusan model sebelum digunakan.
2. Memahami limitasi model.
3. Tidak menggunakan output model secara membuta tuli.
4. Memastikan data yang digunakan mempunyai hak penggunaan yang sesuai.
5. Menjaga privasi individu dalam dataset atau imej.
6. Mendapatkan pengesahan manusia bagi keputusan yang memerlukan pertimbangan profesional.

---

# 15.0 VERSION CONTROL

**GitHub** digunakan untuk mengurus perubahan fail projek melalui **version control**.

Setiap perubahan penting perlu direkodkan menggunakan commit message yang jelas dan profesional.

Contoh commit message:

```text
Initial project setup
Add CNN notebook
Fix CNN input dimension
Fix dataset path
Add YOLO11n object detection
Fix YOLO confidence threshold
Add confidence threshold experiment
Add video inference
Update README documentation
```

Amalan ini membantu menjejaki perubahan yang dilakukan sepanjang proses pembangunan projek.

---

# 16.0 KESIMPULAN

Projek ini berjaya menggabungkan pembangunan model **Computer Vision menggunakan CNN dan YOLO11n** dengan pengurusan projek melalui GitHub.

Proses pembetulan kod, pengujian model, eksperimen confidence threshold dan video inference telah didokumentasikan secara sistematik.

Selain aspek teknikal, projek ini turut menekankan kepentingan **Responsible AI** melalui penghormatan terhadap hak cipta, pengurusan privasi data, dokumentasi limitasi model dan tanggungjawab pengguna.

Dokumentasi ini disediakan sebagai rujukan kepada jurutera atau pengguna lain supaya proses pembangunan dan penggunaan model Computer Vision dapat dilakukan secara lebih sistematik, selamat, beretika dan bertanggungjawab.

---

## PROJECT INFORMATION

| Perkara          | Maklumat                           |
| ---------------- | ---------------------------------- |
| Kursus           | DKA3223 – AI Untuk Computer Vision |
| Projek           | Computer Vision                    |
| Model            | CNN & YOLO11n                      |
| Bahasa           | Python                             |
| Platform         | Google Colab                       |
| Framework        | PyTorch                            |
| Object Detection | YOLO11n                            |
| GPU              | NVIDIA Tesla T4                    |
| Version Control  | GitHub                             |
| Video Inference  | 120 Frames                         |
| Repository       | Private Repository                 |

---

## STATUS PROJEK

| Komponen                        | Status      |
| ------------------------------- | ----------- |
| CNN Simulation                  | ✅ Completed |
| CNN Code Correction             | ✅ Completed |
| YOLO11n Object Detection        | ✅ Completed |
| Confidence Threshold Experiment | ✅ Completed |
| Video Inference                 | ✅ Completed |
| Responsible AI Documentation    | ✅ Completed |
| Version Control                 | ✅ Completed |
| README Documentation            | ✅ Completed |

---

**DKA3223 – AI Untuk Computer Vision**

> Projek ini dibangunkan bagi tujuan pembelajaran, pengujian dan dokumentasi teknologi Artificial Intelligence dan Computer Vision.
