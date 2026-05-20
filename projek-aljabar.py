# ============================================
# TUGAS PCA PADA CITRA DIGITAL
# KOMPRESI CITRA MENGGUNAKAN PCA
# ============================================

# =========================
# 1. IMPORT LIBRARY
# =========================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

# =========================
# 2. MEMBACA CITRA
# =========================

# Link gambar
url_gambar1 = "https://duanol.unnes.ac.id/v2/primer/user_ava/2504130027"
url_gambar2 = "https://tse4.mm.bing.net/th/id/OIP.-4agtzeVGvM8KsQECvWDHQHaIP?rs=1&pid=ImgDetMain&o=7&rm=3"


# Mengambil gambar dari internet
response = requests.get(url_gambar1)
#mengambil gambar buat dibandingkan
img1 = Image.open(BytesIO(requests.get(url_gambar1).content)).convert("L")
img2 = Image.open(BytesIO(requests.get(url_gambar1).content)).convert("L")


# Membuka gambar
gambar = Image.open(BytesIO(response.content))

print("=== INFORMASI CITRA ===")
print("Mode Warna :", gambar.mode)
print("Ukuran :", gambar.size)

# =========================
# 3. EDA AWAL (Exploratory Data Analysis)
# =========================

# Menampilkan gambar asli
plt.figure(figsize=(6,6))
plt.imshow(gambar)
plt.title("Gambar Asli")
plt.axis("off")
plt.show()

# Konversi ke grayscale
gambar_gray = gambar.convert("L")

# Histogram citra
plt.figure(figsize=(6,4))
plt.hist(np.array(gambar_gray).ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Intensitas Awal")
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")
plt.show()

# =========================
# 4. MENGUBAH CITRA MENJADI MATRIKS
# =========================

matriks_gambar = np.array(gambar_gray)

print("\nUkuran Matriks :", matriks_gambar.shape)

# Ubah ke float
matriks_gambar = matriks_gambar.astype(float)

# Menampilkan grayscale
plt.figure(figsize=(6,6))
plt.imshow(matriks_gambar, cmap='gray')
plt.title("Citra Grayscale")
plt.axis("off")
plt.show()

# =========================
# 5. NORMALISASI / CENTERING
# =========================

# Normalisasi 0-1
matriks_normal = matriks_gambar / 255.0

# Mean tiap kolom
mean = np.mean(matriks_normal, axis=0)

# Centering data
data_centered = matriks_normal - mean

print("\nData berhasil dinormalisasi dan dicentering")

# =========================
# 6. MENGHITUNG MATRIKS KOVARIANS
# =========================

kovarians = np.cov(data_centered,
                   rowvar=False)

print("Ukuran Matriks Kovarians :",
      kovarians.shape)

# =========================
# 7. EIGENVALUE & EIGENVECTOR
# =========================

eigen_values, eigen_vectors = np.linalg.eigh(kovarians)

print("\nEigenvalue berhasil dihitung")

# =========================
# 8. MENGURUTKAN EIGENVALUE
#    DAN EIGENVECTOR
# =========================

urutan = np.argsort(eigen_values)[::-1]

eigen_values = eigen_values[urutan]
eigen_vectors = eigen_vectors[:, urutan]

print("\n10 Eigenvalue Terbesar")
print(eigen_values[:10])

# Grafik eigenvalue
plt.figure(figsize=(8,4))
plt.plot(eigen_values[:30], marker='o')
plt.title("Grafik Eigenvalue PCA")
plt.xlabel("Komponen")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()

# =========================
# 9. MEMILIH JUMLAH
#    KOMPONEN UTAMA
# =========================

explained_variance_ratio = (
    eigen_values / np.sum(eigen_values)
)

cumulative_variance = np.cumsum(
    explained_variance_ratio
)

# jumlah komponen(K)
jumlah_komponen = 50

print("\nJumlah Komponen Dipilih :",
      jumlah_komponen)

print("Variansi Kumulatif :",
      cumulative_variance[jumlah_komponen-1])

# Grafik variansi kumulatif
plt.figure(figsize=(8,4))
plt.plot(cumulative_variance, marker='o')
plt.title("Variansi Kumulatif PCA")
plt.xlabel("Jumlah Komponen")
plt.ylabel("Variansi")
plt.grid()
plt.show()

# =========================
# 10. PROYEKSI DATA
# =========================

komponen_utama = eigen_vectors[:, :jumlah_komponen]

hasil_pca = np.dot(
    data_centered,
    komponen_utama
)

print("\nUkuran Hasil Proyeksi :",
      hasil_pca.shape)

# =========================
# 11. REKONSTRUKSI CITRA
# =========================

rekonstruksi = np.dot(
    hasil_pca,
    komponen_utama.T
) + mean

# Kembali ke 0-255
rekonstruksi = rekonstruksi * 255

# Batasi piksel
rekonstruksi = np.clip(
    rekonstruksi,
    0,
    255
)

rekonstruksi = rekonstruksi.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar,
           cmap='gray')
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi,
           cmap='gray')
plt.title(
    f"Hasil PCA ({jumlah_komponen} Komponen)"
)
plt.axis("off")

plt.show()

# =========================
# 12. EVALUASI KUALITAS
#    HASIL KOMPRESI
# =========================

# Mean Squared Error (MSE)
mse = np.mean(
    (matriks_gambar - rekonstruksi) ** 2
)

# Peak Signal to Noise Ratio (PSNR)
if mse == 0:
    psnr = 100
else:
    psnr = (
        20 * np.log10(255 /
        np.sqrt(mse))
    )

print("\n=== EVALUASI KOMPRESI ===")
print("MSE :", mse)
print("PSNR :", psnr, "dB")

# =========================
# PERBANDINGAN FOTO &
# PERSENTASE KEMIRIPAN
# =========================

# Menghitung SSIM
nilai_ssim = ssim(
    matriks_gambar.astype(np.uint8),
    rekonstruksi
)

# Persentase kemiripan
persentase_kemiripan = nilai_ssim * 100

print("\n=== PERBANDINGAN CITRA ===")
print("SSIM :", nilai_ssim)
print("Kemiripan :", persentase_kemiripan, "%")

# Menampilkan perbandingan
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar, cmap='gray')
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi, cmap='gray')
plt.title(
    f"Hasil PCA\nKemiripan: {persentase_kemiripan:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Selisih gambar (difference image)
selisih = np.abs(
    matriks_gambar.astype(np.float32)
    - rekonstruksi.astype(np.float32)
)

plt.figure(figsize=(6,6))
plt.imshow(selisih, cmap='hot')
plt.title("Perbedaan Asli vs Kompresi")
plt.colorbar()
plt.axis("off")
plt.show()

# =========================
# 13. EDA SETELAH KOMPRESI
# =========================

# Histogram hasil kompresi
plt.figure(figsize=(6,4))

plt.hist(rekonstruksi.ravel(),
         bins=256,
         color='gray')

plt.title(
    "Histogram Intensitas Setelah PCA"
)
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")
plt.show()

# Perbandingan histogram
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(matriks_gambar.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Asli")

plt.subplot(1,2,2)
plt.hist(rekonstruksi.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Setelah Kompresi")

plt.tight_layout()
plt.show()

# =========================
# INFORMASI AKHIR
# =========================

print("\n=== INFORMASI AKHIR ===")
print("Total Eigenvalue :",
      len(eigen_values))
print("Komponen Dipakai :",
      jumlah_komponen)
print("Variansi Tersimpan :",
      cumulative_variance[
          jumlah_komponen-1
      ])


