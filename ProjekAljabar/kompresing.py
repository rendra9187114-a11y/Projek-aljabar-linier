# ============================================
# TUGAS PCA PADA CITRA DIGITAL
# Menggunakan Gambar dari LINK
# + INFORMASI UKURAN FILE
# ============================================

# =========================
# IMPORT LIBRARY
# =========================

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO

folder_input = "test"
folder_output = "hasil kompresi dan detail"

os.makedirs(folder_output, exist_ok=True)
# =========================
# MEMBACA GAMBAR DARI LINK
# =========================
for file in os.listdir(folder_input):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        file_gambar = os.path.join(folder_input, file)

        nama_dasar = os.path.splitext(file)[0]

        folder_foto = os.path.join(folder_output, nama_dasar)

        os.makedirs(folder_foto, exist_ok=True)

        # membuka
        gambar = Image.open(file_gambar)

        # Ukuran file asli
        gambar.save("gambar_asli.jpg")
        size_asli = os.path.getsize("gambar_asli.jpg")

        # Konversi ukuran file
        if size_asli < 1024 * 1024:
            size_asli_text = f"{size_asli/1024:.2f} KB"
        else:
            size_asli_text = f"{size_asli/(1024*1024):.2f} MB"

        # =========================
        # INFORMASI GAMBAR
        # =========================

        print("=== INFORMASI GAMBAR ===")
        print("Nama File :", file)
        print("Mode Warna :", gambar.mode)
        print("Ukuran Gambar :", gambar.size)
        print("Ukuran File Asli :", size_asli_text)

        # =========================
        # MENAMPILKAN GAMBAR ASLI
        # =========================

        plt.figure(figsize=(6, 6))
        plt.imshow(gambar)
        plt.title(f"Gambar Asli\nSize : {size_asli_text}")
        plt.axis("off")
        plt.show()

        # =========================
        # KONVERSI KE GRAYSCALE
        # =========================

        # Mengubah gambar menjadi grayscale
        gambar_gray = gambar.convert("L")

        # Menyimpan grayscale
        gambar_gray.save("gambar_grayscale.jpg")

        # Ukuran grayscale
        size_gray = os.path.getsize("gambar_grayscale.jpg")

        if size_gray < 1024 * 1024:
            size_gray_text = f"{size_gray/1024:.2f} KB"
        else:
            size_gray_text = f"{size_gray/(1024*1024):.2f} MB"

        print("Ukuran File Grayscale :", size_gray_text)

        # Mengubah menjadi array NumPy
        matriks_gambar = np.array(gambar_gray)

        # Mengubah tipe data menjadi float
        matriks_gambar = matriks_gambar.astype(float)

        print("Ukuran Matriks :", matriks_gambar.shape)

        # =========================
        # MENAMPILKAN GRAYSCALE
        # =========================

        plt.figure(figsize=(6, 6))
        plt.imshow(matriks_gambar, cmap="gray")
        plt.title(f"Gambar Grayscale\nSize : {size_gray_text}")
        plt.axis("off")

        plt.show()

        # =========================
        # NORMALISASI DATA
        # =========================

        matriks_normal = matriks_gambar / 255.0

        # =========================
        # MENGHITUNG MEAN
        # =========================

        mean = np.mean(matriks_normal, axis=0)

        data_centered = matriks_normal - mean

        # =========================
        # MENGHITUNG KOVARIANSI
        # =========================

        kovarians = np.cov(data_centered, rowvar=False)

        print("Ukuran Matriks Kovarians :", kovarians.shape)

        # =========================
        # EIGEN VALUE & EIGEN VECTOR
        # =========================

        eigen_values, eigen_vectors = np.linalg.eigh(kovarians)

        # Mengurutkan eigen value terbesar
        urutan = np.argsort(eigen_values)[::-1]

        eigen_values = eigen_values[urutan]
        eigen_vectors = eigen_vectors[:, urutan]

        # =========================
        # MENAMPILKAN 10 EIGEN VALUE
        # =========================

        print("\n10 Eigen Value Terbesar :")
        print(eigen_values[:10])

        # =========================
        # GRAFIK EIGEN VALUE
        # =========================

        plt.figure(figsize=(10, 5))
        plt.plot(eigen_values[:20], marker="o")
        plt.title("Grafik Eigen Value PCA")
        plt.xlabel("Komponen Utama")
        plt.ylabel("Nilai Eigen")
        plt.grid(True)
        grafik_eigen = os.path.join(folder_foto, "grafik_eigenvalue.png")

        plt.savefig(grafik_eigen, dpi=300, bbox_inches="tight")
        plt.show()

        # =========================
        # VARIANSI KUMULATIF
        # =========================

        explained_variance_ratio = eigen_values / np.sum(eigen_values)

        cumulative_variance = np.cumsum(explained_variance_ratio)

        # =========================
        # GRAFIK VARIANSI KUMULATIF
        # =========================

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_variance, marker="o")
        plt.title("Grafik Variansi Kumulatif PCA")
        plt.xlabel("Jumlah Komponen")
        plt.ylabel("Variansi Kumulatif")
        plt.grid(True)

        grafik_variansi = os.path.join(folder_foto, "grafik_variansi_kumulatif.png")
        plt.savefig(grafik_variansi, dpi=300, bbox_inches="tight")
        plt.show()

        # =========================
        # PCA DENGAN BERBAGAI KOMPONEN
        # =========================

        daftar_komponen = [50, 75, 100, 150, 300]

        plt.figure(figsize=(20, 10))

        for i, jumlah_komponen in enumerate(daftar_komponen):

            # Mengambil komponen utama
            komponen_utama = eigen_vectors[:, :jumlah_komponen]

            # Transformasi PCA
            hasil_pca = np.dot(data_centered, komponen_utama)

            # Rekonstruksi gambar
            rekonstruksi = np.dot(hasil_pca, komponen_utama.T) + mean

            # Mengembalikan ke skala 0-255
            rekonstruksi = rekonstruksi * 255.0

            # Membatasi nilai piksel
            rekonstruksi = np.clip(rekonstruksi, 0, 255)

            # Mengubah ke uint8
            rekonstruksi = rekonstruksi.astype(np.uint8)

            # =========================
            # MENYIMPAN HASIL KOMPRESI
            # =========================

            nama_file = os.path.join(
                folder_foto, f"{nama_dasar}_pca_{jumlah_komponen}.jpg"
            )

            Image.fromarray(rekonstruksi).save(nama_file)

            # Menghitung ukuran file
            size_pca = os.path.getsize(nama_file)

            if size_pca < 1024 * 1024:
                size_text = f"{size_pca/1024:.2f} KB"
            else:
                size_text = f"{size_pca/(1024*1024):.2f} MB"

            print(f"\nPCA {jumlah_komponen} Komponen")
            print("Ukuran File :", size_text)

            # Menampilkan hasil PCA
            plt.subplot(2, 3, i + 1)
            plt.imshow(rekonstruksi, cmap="gray")
            plt.title(f"PCA {jumlah_komponen}\n{size_text}")
            plt.axis("off")

            # Informasi variansi
            if jumlah_komponen <= len(cumulative_variance):

                print("Variansi Kumulatif :", cumulative_variance[jumlah_komponen - 1])

        # =========================
        # MENAMPILKAN GAMBAR ASLI
        # =========================

        plt.subplot(2, 3, 6)
        plt.imshow(matriks_gambar, cmap="gray")
        plt.title(f"Gambar Asli\nSize : {size_asli_text}")
        plt.axis("off")
        plt.tight_layout()

        plt.show()

        # =========================
        # INFORMASI AKHIR
        # =========================

        print("\n=== INFORMASI PCA ===")
        print("Total Eigen Value :", len(eigen_values))
        print(f"Jumlah Komponen Yang Diuji : {len(daftar_komponen)}\n")
