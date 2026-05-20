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
url_gambar1 = "https://i.ibb.co.com/v61fmpw4/foto-kecil-cokelat.jpg"
url_gambar2 = "https://i.ibb.co.com/m5dxs8kd/foto-sekarang.jpg"


# Mengambil gambar dari internet
response1 = requests.get(url_gambar1)
response2 = requests.get(url_gambar2)

# Membuka gambar
gambar1 = Image.open(BytesIO(response1.content))
gambar2 = Image.open(BytesIO(response2.content))

#informasi citra foto 1
print("=== INFORMASI CITRA FOTO 1===")
print("Mode Warna :", gambar1.mode)
print("Ukuran :", gambar1.size)

# =========================
# 3. EDA AWAL (Exploratory Data Analysis)
# =====================0====

# Menampilkan gambar asli
plt.figure(figsize=(10,4))
# Foto 1
plt.subplot(1,2,1)
plt.imshow(gambar1)
plt.title("FOTO ASLI 1")
plt.axis("off")

#informasi citra foto 1
print("\n=== INFORMASI CITRA FOTO 2===")
print("Mode Warna :", gambar2.mode)
print("Ukuran :", gambar2.size)
print("\n")

# Foto 2
plt.subplot(1,2,2)
plt.imshow(gambar2)
plt.title("FOTO ASLI 2")
plt.axis("off")

plt.tight_layout()
plt.show()

# Konversi ke grayscale
gambar1_gray = gambar1.convert("L")
gambar2_gray = gambar2.convert("L")

# Histogram citra
plt.figure(figsize=(10,4))
# Foto 1
plt.subplot(1,2,1)
plt.hist(np.array(gambar1_gray).ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Intensitas Awal Pada Foto 1")
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")


# Foto 1
plt.subplot(1,2,2)
plt.hist(np.array(gambar2_gray).ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Intensitas Awal Pada Foto 2")
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")

plt.tight_layout()
plt.show()

# =========================
# 4. MENGUBAH CITRA MENJADI MATRIKS
# =========================

matriks_gambar1 = np.array(gambar1_gray)
matriks_gambar2 = np.array(gambar2_gray)

print("\nUkuran Matriks Foto 1:", matriks_gambar1.shape)
print("Ukuran Matriks Foto 2 :", matriks_gambar2.shape)
print("\n")

# Ubah ke float
matriks_gambar1 = matriks_gambar1.astype(float)
matriks_gambar2 = matriks_gambar2.astype(float)

# Menampilkan grayscale
plt.figure(figsize=(10,4))
#foto 1
plt.subplot(1,2,1)
plt.imshow(matriks_gambar1, cmap='gray')
plt.title("Citra Grayscale Foto 1")
plt.axis("off")

#foto 2
plt.subplot(1,2,2)
plt.imshow(matriks_gambar2, cmap='gray')
plt.title("Citra Grayscale Foto 2")
plt.axis("off")

plt.tight_layout()
plt.show()

# =========================
# 5. NORMALISASI / CENTERING
# =========================
  
# Normalisasi 0-1 padafoto 1
matriks_normal1 = matriks_gambar1 / 255.0

# Normalisasi 0-1 padafoto 2
matriks_normal2 = matriks_gambar2 / 255.0

# Mean tiap kolom
mean1 = np.mean(matriks_normal1, axis=0)
mean2 = np.mean(matriks_normal2, axis=0)

# Centering data
data_centered1 = matriks_normal1 - mean1
data_centered2 = matriks_normal2 - mean2

print("\nData berhasil dinormalisasi dan dicentering")

# =========================
# 6. MENGHITUNG MATRIKS KOVARIANS
# =========================

kovarians1 = np.cov(data_centered1, rowvar=False)
kovarians2 = np.cov(data_centered2, rowvar=False)

print("Ukuran Matriks Kovarians Foto 1:",kovarians1.shape)
print("Ukuran Matriks Kovarians Foto 1:",kovarians2.shape)

# =========================
# 7. EIGENVALUE & EIGENVECTOR
# =========================

eigen_values1, eigen_vectors1 = np.linalg.eigh(kovarians1)
eigen_values2, eigen_vectors2 = np.linalg.eigh(kovarians2)

print("\nEigenvalue berhasil dihitung")

# =========================
# 8. MENGURUTKAN EIGENVALUE
#    DAN EIGENVECTOR
# =========================
# Foto 1
urutan1 = np.argsort(eigen_values1)[::-1]

eigen_values1 = eigen_values1[urutan1]
eigen_vectors1 = eigen_vectors1[:, urutan1]

# Foto 2
urutan2 = np.argsort(eigen_values2)[::-1]


eigen_values2 = eigen_values2[urutan2]
eigen_vectors2 = eigen_vectors2[:, urutan2]

print("\n10 Eigenvalue Terbesar dari Foto 1:")
print(eigen_values1[:10])

# Grafik eigenvalue
plt.figure(figsize=(8,4))
plt.plot(eigen_values1[:30], marker='o')
plt.title("Grafik Eigenvalue PCA")
plt.xlabel("Komponen")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()

print("\n10 Eigenvalue Terbesar dari Foto 2:")
print(eigen_values2[:10])

# Grafik eigenvalue
plt.figure(figsize=(8,4))
plt.plot(eigen_values2[:30], marker='o')
plt.title("Grafik Eigenvalue PCA")
plt.xlabel("Komponen")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()

# =========================
# 9. MEMILIH JUMLAH
#    KOMPONEN UTAMA
# =========================

explained_variance_ratio1 = (eigen_values1 / np.sum(eigen_values1))
explained_variance_ratio2 = (eigen_values2 / np.sum(eigen_values2))

cumulative_variance1 = np.cumsum(explained_variance_ratio1)
cumulative_variance2 = np.cumsum(explained_variance_ratio2)

# =========================
# =========================
# jumlah komponen(K)
# jumlah komponen(K)
# jumlah komponen(K)
# jumlah komponen(K)
# =========================
# =========================
jumlah_komponen = 1

print("\nJumlah Komponen Dipilih :",
      jumlah_komponen)

print("Variansi Kumulatif Foto 1:", cumulative_variance1[jumlah_komponen-1])

# Grafik variansi kumulatif
plt.figure(figsize=(8,4))
plt.plot(cumulative_variance1, marker='o')
plt.title("Variansi Kumulatif PCA Pada Foto 1")
plt.xlabel("Jumlah Komponen")
plt.ylabel("Variansi")
plt.grid()
plt.show()

print("Variansi Kumulatif Foto 2:", cumulative_variance2[jumlah_komponen-1])

# Grafik variansi kumulatif
plt.figure(figsize=(8,4))
plt.plot(cumulative_variance2, marker='o')
plt.title("Variansi Kumulatif PCA Pada Foto 2")
plt.xlabel("Jumlah Komponen")
plt.ylabel("Variansi")
plt.grid()
plt.show()

# =========================
# 10. PROYEKSI DATA
# =========================

komponen_utama1 = eigen_vectors1[:, :jumlah_komponen]
komponen_utama2 = eigen_vectors2[:, :jumlah_komponen]

hasil_pca1 = np.dot(data_centered1,komponen_utama1)
hasil_pca2 = np.dot(data_centered2,komponen_utama2)

print("\nUkuran Hasil Proyeksi :",hasil_pca1.shape)
print("\nUkuran Hasil Proyeksi :",hasil_pca2.shape)
print("\n")

# =========================
# 11. REKONSTRUKSI CITRA
# =========================

#Foto 1
rekonstruksi1 = np.dot(
    hasil_pca1,
    komponen_utama1.T
) + mean1

# Kembali ke 0-255
rekonstruksi1 = rekonstruksi1 * 255

# Batasi piksel
rekonstruksi1 = np.clip(
    rekonstruksi1,
    0,
    255
)

rekonstruksi1 = rekonstruksi1.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar1,
           cmap='gray')
plt.title("Citra Asli Foto 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi1,
           cmap='gray')
plt.title(
    f"Hasil PCA ({jumlah_komponen} Komponen)"
)
plt.axis("off")

plt.show()

#Foto 2
rekonstruksi2 = np.dot(
    hasil_pca2,
    komponen_utama2.T
) + mean2

# Kembali ke 0-255
rekonstruksi2 = rekonstruksi2 * 255

# Batasi piksel
rekonstruksi2 = np.clip(
    rekonstruksi2,
    0,
    255
)

rekonstruksi2 = rekonstruksi2.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar2,
           cmap='gray')
plt.title("Citra Asli Foto 2")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi2,
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
#foto1
# Mean Squared Error (MSE)
mse1 = np.mean(
    (matriks_gambar1 - rekonstruksi1) ** 2
)

# Peak Signal to Noise Ratio (PSNR)
if mse1 == 0:
    psnr1 = 100
else:
    psnr1 = (
        20 * np.log10(255 /
        np.sqrt(mse1))
    )

print("\n=== EVALUASI KOMPRESI FOTO 1 ===")
print("MSE :", mse1)
print("PSNR :", psnr1, "dB")

#foto2
# Mean Squared Error (MSE)
mse2 = np.mean(
    (matriks_gambar2 - rekonstruksi2) ** 2
)

# Peak Signal to Noise Ratio (PSNR)
if mse2 == 0:
    psnr2 = 100
else:
    psnr2 = (
        20 * np.log10(255 /
        np.sqrt(mse2))
    )

print("\n=== EVALUASI KOMPRESI FOTO 2 ===")
print("MSE :", mse2)
print("PSNR :", psnr2, "dB")

# =========================
# PERBANDINGAN FOTO &
# PERSENTASE KEMIRIPAN
# =========================

#foto1
# Menghitung SSIM
nilai_ssim1 = ssim(
    matriks_gambar1.astype(np.uint8),
    rekonstruksi1
)

# Persentase kemiripan
persentase_kemiripan1 = nilai_ssim1 * 100

print("\n=== PERBANDINGAN CITRA FOTO 1===")
print("SSIM :", nilai_ssim1)
print("Kemiripan :", persentase_kemiripan1, "%")

# Menampilkan perbandingan
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar1, cmap='gray')
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi1, cmap='gray')
plt.title(
    f"Hasil PCA\nKemiripan: {persentase_kemiripan1:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Selisih gambar (difference image)
selisih1 = np.abs(
    matriks_gambar1.astype(np.float32)
    - rekonstruksi1.astype(np.float32)
)

plt.figure(figsize=(6,6))
plt.imshow(selisih1, cmap='hot')
plt.title("Perbedaan Asli vs Kompresi")
plt.colorbar()
plt.axis("off")
plt.show()

#foto 2
# Menghitung SSIM
nilai_ssim2 = ssim(
    matriks_gambar2.astype(np.uint8),
    rekonstruksi2
)

# Persentase kemiripan
persentase_kemiripan2 = nilai_ssim2 * 100

print("\n=== PERBANDINGAN CITRA FOTO 2 ===")
print("SSIM :", nilai_ssim2)
print("Kemiripan :", persentase_kemiripan2, "%")

# Menampilkan perbandingan
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(matriks_gambar2, cmap='gray')
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rekonstruksi2, cmap='gray')
plt.title(
    f"Hasil PCA\nKemiripan: {persentase_kemiripan2:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Selisih gambar (difference image)
selisih2 = np.abs(
    matriks_gambar2.astype(np.float32)
    - rekonstruksi2.astype(np.float32)
)

plt.figure(figsize=(6,6))
plt.imshow(selisih2, cmap='hot')
plt.title("Perbedaan Asli vs Kompresi")
plt.colorbar()
plt.axis("off")
plt.show()


# =========================
# FOTO ASLI 1 vs FOTO ASLI 2
# =========================

# Menghitung SSIM
nilai_ssim_asli = ssim(
    matriks_gambar1.astype(np.uint8),
    matriks_gambar2.astype(np.uint8)
)

# Persentase
persentase_asli = (
    nilai_ssim_asli * 100
)

print(
    "\n=== FOTO ASLI 1 vs FOTO ASLI 2 ==="
)
print(
    "SSIM :",
    nilai_ssim_asli
)
print(
    "Kemiripan :",
    persentase_asli,
    "%"
)

# Tampilkan foto
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(
    matriks_gambar1,
    cmap='gray'
)
plt.title("Foto Asli 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(
    matriks_gambar2,
    cmap='gray'
)
plt.title(
    f"Foto Asli 2\nKemiripan {persentase_asli:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Difference image
selisih_asli = np.abs(
    matriks_gambar1.astype(np.float32)
    -
    matriks_gambar2.astype(np.float32)
)

plt.figure(figsize=(6,6))
plt.imshow(
    selisih_asli,
    cmap='hot'
)
plt.title(
    "Perbedaan Foto Asli 1 vs 2"
)
plt.colorbar()
plt.axis("off")
plt.show()

# =========================
# FOTO KOMPRESI 1 vs 2
# =========================

# Menghitung SSIM
nilai_ssim_kompresi = ssim(
    rekonstruksi1,
    rekonstruksi2
)

# Persentase
persentase_kompresi = (
    nilai_ssim_kompresi * 100
)

print(
    "\n=== FOTO KOMPRESI 1 vs FOTO KOMPRESI 2 ==="
)
print(
    "SSIM :",
    nilai_ssim_kompresi
)
print(
    "Kemiripan :",
    persentase_kompresi,
    "%"
)

# Menampilkan foto
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(
    rekonstruksi1,
    cmap='gray'
)
plt.title("Kompresi 1")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(
    rekonstruksi2,
    cmap='gray'
)
plt.title(
    f"Kompresi 2\nKemiripan {persentase_kompresi:.2f}%"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Difference image
selisih_kompresi = np.abs(
    rekonstruksi1.astype(np.float32)
    -
    rekonstruksi2.astype(np.float32)
)

plt.figure(figsize=(6,6))
plt.imshow(
    selisih_kompresi,
    cmap='hot'
)
plt.title(
    "Perbedaan Kompresi 1 vs 2"
)
plt.colorbar()
plt.axis("off")
plt.show()

# =========================
# 13. EDA SETELAH KOMPRESI
# =========================

#foto1
# Histogram hasil kompresi
print("")
plt.figure(figsize=(1,1))
plt.title("FOTO 1")
plt.axis("off")
plt.show()

plt.figure(figsize=(6,4))

plt.hist(rekonstruksi1.ravel(),
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
plt.hist(matriks_gambar1.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Asli")

plt.subplot(1,2,2)
plt.hist(rekonstruksi1.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Setelah Kompresi")

plt.tight_layout()
plt.show()

#foto2
# Histogram hasil kompresi
print("\n")
plt.figure(figsize=(1,1))
plt.title("FOTO 2")
plt.axis("off")
plt.show()

plt.figure(figsize=(6,4))

plt.hist(rekonstruksi1.ravel(),
         bins=256,
         color='gray')

plt.title(
    "Histogram Intensitas Setelah PCA "
)
plt.xlabel("Nilai Piksel")
plt.ylabel("Frekuensi")
plt.show()

# Perbandingan histogram
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(matriks_gambar2.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Asli")

plt.subplot(1,2,2)
plt.hist(rekonstruksi2.ravel(),
         bins=256,
         color='gray')
plt.title("Histogram Setelah Kompresi")

plt.tight_layout()
plt.show()

# =========================
# INFORMASI AKHIR
# =========================
#foto 1
print("\n=== INFORMASI AKHIR FOTO 1 ===")
print("Total Eigenvalue :",
      len(eigen_values1))
print("Komponen Dipakai :",
      jumlah_komponen)
print("Variansi Tersimpan :",
      cumulative_variance1[
          jumlah_komponen-1
      ])

#foto 2
print("\n=== INFORMASI AKHIR FOTO 2 ===")
print("Total Eigenvalue :",
      len(eigen_values2))
print("Komponen Dipakai :",
      jumlah_komponen)
print("Variansi Tersimpan :",
      cumulative_variance2[
          jumlah_komponen-1
      ])

