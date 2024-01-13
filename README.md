## Sistem Manajemen Parkir SMK Kristen Immanuel  

Merupakan program manajemen parkir untuk SMK Kristen Immanuel yang dibuat menggunakan bahasa python. Program ini dapat mengenali dan mengidentifikasi pemilik kendaraan berdasarkan gambar yang tertangkap melalui kamera. Algoritma yang digunakan untuk mengklasifikasi / mengenali karakter pada plat yang digunakan adalah Naive Bayes.  

Adapun fitur yang terdapat pada aplikasi ini antara lain :  
- mengidentifikasi kendaraan dan pemilik kendaraan berdasarkan plat nomor kendaraan  
- menyediakan informasi terkait parkir seperti jumlah kendaraan, ketersediaan lahan parkir, waktu keluar masuknya kendaraan  

Proses pengenalan plat nomor kendaraaan memiliki tahapan sebagai berikut :  
-	Konversi grayscale menggunakan metode luminosity
- Noise removal menggunakan metode gaussian blur
- Peningkatan kontras menggunakan metode white top-hat 
- Mengubah data gambar menjadi data biner (binerisasi) menggunakan metode fixed thresholding
- Segmentasi karakter menggunakan proyeksi profil histogram
- Ekstraksi ciri tepi menggunakan canny edge detection
- Klasifikasi karakter menggunakan algoritma naive bayes
