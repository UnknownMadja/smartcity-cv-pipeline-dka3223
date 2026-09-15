# DKA3223 - AI UNTUK COMPUTER VISION

## 📌 Maklumat Projek

| Maklumat | Butiran |
|---|---|
| Kursus | DKA3223 - AI Untuk Computer Vision |
| Program | Teknologi Komputeran |
| Semester | Semester 3 DVM |
| Nama | AMIR ADAM BIN AMIRUDDIN |
| Angka Giliran | PKV0125KA001 |
| Persekitaran | Google Colab |
| Platform | GitHub |
| Model | CNN & YOLO11n |

---

## 📖 1.0 Pengenalan

Projek ini dibangunkan bagi kursus **DKA3223 - AI Untuk Computer Vision**. Projek ini memfokuskan kepada pembangunan dan dokumentasi model Computer Vision menggunakan **Convolutional Neural Network (CNN)** dan **YOLO11n Object Detection**.

Projek ini dilaksanakan menggunakan Google Colab sebagai persekitaran pembangunan dan GitHub sebagai platform pengurusan repositori. Tujuan utama projek adalah untuk mengurus fail projek secara sistematik, mendokumentasikan pembetulan kod, menjalankan eksperimen model serta mengamalkan prinsip **Responsible AI**.

---

## 🎯 2.0 Objektif Projek

Objektif projek ini adalah:

- Membangunkan model Computer Vision menggunakan Google Colab.
- Melaksanakan simulasi Convolutional Neural Network (CNN).
- Melaksanakan pengesanan objek menggunakan YOLO11n.
- Membetulkan ralat dan `FIX ME` dalam kod.
- Menjalankan eksperimen terhadap confidence threshold.
- Melaksanakan inferens imej dan video.
- Mengamalkan version control menggunakan GitHub.
- Mendokumentasikan aspek etika, privasi, hak cipta dan limitasi model.

---

## 💻 3.0 Persekitaran Pembangunan

Persekitaran yang digunakan dalam projek:

```text
Google Colab
Python
PyTorch
YOLO11n
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

GitHub
CUDA GPU
Tesla T4
🧠 5.0 Convolutional Neural Network (CNN)

Simulasi CNN digunakan bagi proses klasifikasi imej. Kod telah diperiksa dan beberapa pembetulan dilakukan bagi memastikan notebook dapat dijalankan dengan baik.

Pembetulan yang dilakukan:
5.1 Dataset Path

Laluan dataset diperbaiki supaya lebih fleksibel. Folder train dan test dibina menggunakan os.path.join() dan kewujudannya disemak sebelum dataset dimuatkan.

5.2 GPU dan Reproducibility

GPU diperiksa bagi memastikan peranti yang digunakan dapat dikenal pasti.

Kod berikut digunakan:

torch.cuda.manual_seed_all(42)
torch.cuda.get_device_name(0)

Penggunaan seed membantu meningkatkan kebolehulangan eksperimen apabila latihan menggunakan CUDA.

5.3 Visualisasi Dataset

Oleh kerana imej telah melalui proses normalization, proses denormalization digunakan sebelum imej dipaparkan.

image = torch.clamp(image, 0, 1)

Ini membantu memastikan nilai piksel berada dalam julat yang sesuai untuk visualisasi.

5.4 Pembinaan Model CNN

Masalah saiz input bagi fully connected layer telah diperbaiki dengan menggunakan:

nn.AdaptiveAvgPool2d((1, 1))

Kaedah ini menghasilkan feature map bersaiz 64 × 1 × 1 sebelum proses Flatten().

5.5 Bilangan Kelas

Bilangan kelas tidak ditetapkan secara manual. Sebaliknya, bilangan kelas diperoleh daripada dataset:

len(train_dataset.classes)

Pendekatan ini membolehkan model menyesuaikan output berdasarkan jumlah kelas sebenar.

5.6 Confidence Score

Softmax digunakan bagi mendapatkan kebarangkalian setiap kelas. Nilai probability tertinggi digunakan sebagai confidence score dan dipaparkan dalam bentuk peratus.

🎯 6.0 YOLO11n Object Detection

Projek ini turut menggunakan YOLO11n untuk melakukan pengesanan objek pada imej dan video.

Model yang digunakan:

YOLO11n
Model file: yolo11n.pt
Device: GPU
GPU: Tesla T4
🔧 6.1 Pembetulan FIX ME

Sebanyak lima FIX ME telah diselesaikan dalam kod YOLO11n.

FIX ME 1 - Confidence Threshold
CONF_THRESHOLD = 0.25

Nilai ini digunakan untuk menentukan tahap keyakinan minimum sesuatu objek sebelum diterima sebagai hasil pengesanan.

FIX ME 2 - IoU Threshold
IOU_THRESHOLD = 0.45

Nilai IoU digunakan dalam proses Non-Maximum Suppression (NMS) untuk menguruskan bounding box yang bertindih.

FIX ME 3 - Confidence Values
CONF_VALUES = [0.25, 0.45, 0.65, 0.85]

Empat nilai confidence digunakan bagi menjalankan eksperimen dan melihat kesan perubahan threshold terhadap jumlah objek yang dikesan.

FIX ME 4 - Object Class Field
OBJECT_CLASS_FIELD = "nama_kelas"

Kunci nama_kelas digunakan untuk mendapatkan nama kelas objek daripada detection_records.

FIX ME 5 - Maximum Frames
MAX_FRAMES = 120

Jumlah frame maksimum ditetapkan kepada 120 bagi mengawal tempoh pemprosesan video dan penggunaan sumber GPU/RAM.

📊 7.0 Keputusan Object Detection

Model YOLO11n berjaya mengesan sebanyak 5 objek dalam salah satu imej ujian.

Objek yang dikesan:
No.	Kelas	Confidence
1	bus	94.02%
2	person	88.82%
3	person	87.83%
4	person	85.58%
5	person	62.19%
Jumlah objek mengikut kelas:
person : 4
bus    : 1

Jumlah keseluruhan : 5
📈 8.0 Eksperimen Confidence Threshold

Eksperimen dijalankan menggunakan empat nilai confidence threshold.

Threshold	Jumlah Objek	Analisis
0.25	5	Mengesan objek utama dan objek separa terlindung
0.45	5	Masih mengesan objek dengan keyakinan mencukupi
0.65	4	Sebahagian objek dengan keyakinan lebih rendah ditapis
0.85	4	Objek dengan keyakinan tinggi sahaja diluluskan

Peningkatan confidence threshold menyebabkan sesetengah objek dengan nilai keyakinan lebih rendah ditapis daripada keputusan pengesanan.

🎥 9.0 Video Inference

YOLO11n turut digunakan untuk proses inferens video.

Maklumat pemprosesan:

Jumlah frame diproses : 120
Video output          : kv_tron_detection_output.mp4

Jumlah detection sepanjang video:

person : 597
car    : 120
truck  : 120
dog    : 21

Video berjaya diproses selepas semua lima FIX ME diselesaikan.

🔐 10.0 Etika dan Responsible AI

Projek ini mengambil kira prinsip Responsible AI dan etika profesional dalam pembangunan Computer Vision.

Antara amalan yang digunakan ialah:

Menghormati hak cipta dan harta intelek.
Menyatakan sumber kod dan dataset yang digunakan.
Tidak menganggap kod atau dataset pihak lain sebagai hasil sendiri.
Menjaga privasi data.
Tidak mendedahkan maklumat peribadi atau maklumat sensitif.
Mendokumentasikan limitasi model.
Memastikan pengguna memahami bahawa keputusan model tidak semestinya 100% tepat.
Menggalakkan semakan manusia terhadap hasil AI.
🔒 11.0 Privasi dan Keselamatan Data

Repositori projek dikonfigurasikan sebagai Private Repository bagi mengawal akses kepada fail projek.

Maklumat sensitif seperti:

Password
API Key
Access Token
Personal Information
Credentials

tidak sepatutnya dimasukkan ke dalam repositori GitHub.

Sekiranya dataset mengandungi imej manusia atau maklumat yang boleh mengenal pasti individu, data tersebut perlu dikendalikan dengan berhati-hati dan mengikut kebenaran serta tujuan penggunaan yang sesuai.

⚠️ 12.0 Limitasi Model

Model Computer Vision mempunyai beberapa limitasi.

Prestasi model boleh dipengaruhi oleh:

Kualiti dataset.
Jumlah dan kepelbagaian data latihan.
Pencahayaan imej.
Saiz objek.
Kedudukan objek.
Objek yang terlindung.
Perubahan keadaan persekitaran.
Confidence threshold yang digunakan.

Confidence score bukan jaminan bahawa sesuatu ramalan adalah benar. Nilai tersebut menunjukkan tahap keyakinan model terhadap sesuatu pengesanan atau klasifikasi.

👤 13.0 Tanggungjawab Pengguna

Pengguna bertanggungjawab memastikan model digunakan secara beretika.

Pengguna perlu:

Menyemak output model sebelum digunakan.
Memahami limitasi model.
Tidak bergantung sepenuhnya kepada keputusan AI.
Menjaga privasi data.
Menghormati hak cipta dataset dan kod.
Memastikan penggunaan model sesuai dengan tujuan pembangunan.

Bagi keputusan yang penting atau berisiko tinggi, semakan manusia perlu dilakukan sebelum keputusan akhir dibuat.

📝 14.0 Version Control

GitHub digunakan sebagai platform version control bagi menyimpan dan mengurus perubahan projek.

Contoh commit message yang digunakan:

Initial project setup
Add CNN notebook
Fix CNN input dimension
Fix dataset path
Add YOLO11n object detection
Fix YOLO confidence threshold
Fix YOLO IoU threshold
Add confidence threshold experiment
Add video inference
Update README documentation

Commit message yang jelas membantu proses penjejakan perubahan dan memudahkan penyelenggaraan projek.

📚 15.0 Sumber Kod dan Dataset

Kod dan dataset yang digunakan dalam projek perlu diberikan atribusi kepada sumber asal.

Kod sumber:

[MASUKKAN SUMBER KOD SEBENAR DI SINI]

Dataset:

[MASUKKAN SUMBER DATASET SEBENAR DI SINI]

Sebarang penggunaan kod, dataset atau model daripada pihak lain hendaklah mematuhi lesen dan syarat penggunaan yang ditetapkan oleh pemilik asal.

⚠️ Jangan masukkan sumber yang tidak benar. Gunakan sumber sebenar yang digunakan dalam notebook projek.

📌 16.0 Kesimpulan

Projek ini berjaya menggabungkan pembangunan model Computer Vision menggunakan CNN dan YOLO11n dengan pengurusan repositori GitHub secara sistematik.

Bagi CNN, beberapa pembetulan telah dilakukan termasuk pengurusan dataset path, GPU, visualisasi imej, saiz input model, bilangan kelas dan confidence score.

Bagi YOLO11n pula, lima FIX ME telah diselesaikan melibatkan confidence threshold, IoU threshold, confidence values, object class field dan maximum frames. Model juga berjaya digunakan untuk pengesanan objek pada imej serta pemprosesan video.

Selain aspek teknikal, projek ini memberi penekanan kepada Responsible AI, hak cipta, privasi data, limitasi model dan tanggungjawab pengguna. Pendekatan ini membantu memastikan pembangunan dan penggunaan Computer Vision dilakukan secara lebih sistematik, beretika dan bertanggungjawab.
