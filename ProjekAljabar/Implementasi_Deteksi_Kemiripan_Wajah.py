import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# =========================
# kompresi gambar dengan PCA-SVD
# =========================

folder_input = "test"  # folder input untuk gambar yang akan diuji
folder_output = "hasil_kompresi"  # folder output untuk menyimpan hasil kompresi

os.makedirs(folder_output, exist_ok=True)

for file in os.listdir(folder_input):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        file_gambar = os.path.join(folder_input, file)

        gambar = Image.open(file_gambar).convert("L")

        matriks_gambar = np.array(gambar).astype(float)

        matriks_normal = matriks_gambar / 255.0

        mean = np.mean(matriks_normal, axis=0)

        data_centered = matriks_normal - mean

        kovarians = np.cov(data_centered, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(kovarians)

        urutan = np.argsort(eigen_values)[::-1]

        eigen_vectors = eigen_vectors[:, urutan]

        jumlah_komponen = 50

        komponen_utama = eigen_vectors[:, :jumlah_komponen]

        hasil_pca = np.dot(data_centered, komponen_utama)

        rekonstruksi = np.dot(hasil_pca, komponen_utama.T) + mean

        rekonstruksi = np.clip(rekonstruksi * 255, 0, 255).astype(np.uint8)

        size_asli = os.path.getsize(file_gambar)

        if size_asli < 1024 * 1024:
            size_text = f"{size_asli/1024:.2f} KB"
        else:
            size_text = f"{size_asli/(1024*1024):.2f} MB"

        output_file = os.path.join(folder_output, file)

        Image.fromarray(rekonstruksi).save(output_file)

        size_kompresi = os.path.getsize(output_file)

        if size_kompresi < 1024 * 1024:
            size_kompresi_text = f"{size_kompresi/1024:.2f} KB"
        else:
            size_kompresi_text = f"{size_kompresi/(1024*1024):.2f} MB"

        pengurangan = (size_asli - size_kompresi) / size_asli * 100

        print("=========================")
        print(f"Memprose File : {file}")
        print(f"Ukuran Sebelum dikompresi : {size_text}")
        print(f"Ukuran Setelah dikompresi : {size_kompresi_text}")
        print(f"Persentase Pengurangan : {pengurangan:.2f}%")
        print(f"File disimpan di : {output_file}\n")

# =========================
# DETEKSI KEMIRIPAN DENGAN PCA
# =========================
IMG_SIZE = (100, 100)  # ukuran gambar yang akan digunakan untuk deteksi kemiripan


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"\nGagal membaca gambar: {img_path}")
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize
    gray = cv2.resize(gray, IMG_SIZE)
    # flatten
    vector = gray.flatten()
    return vector


def load_dataset(dataset_path):
    data = []
    labels = []
    filenames = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for file in os.listdir(person_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_folder, file)
                vector = preprocess_image(img_path)
                data.append(vector)
                labels.append(person_name)
                filenames.append(file)

    return np.array(data), labels, filenames


def recognize_cosine_topk(test_feature, database_features, labels, filenames, k=3):

    results = []

    test_feature = test_feature.reshape(1, -1)
    for feature, label, file in zip(database_features, labels, filenames):
        feature = feature.reshape(1, -1)
        score = cosine_similarity(test_feature, feature)[0][0]
        results.append((label, file, score))
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:k]


def center_data(X):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    return X_centered, mean_face


def compute_pca_svd(X_centered, num_components=50):
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    eigenfaces = VT[:num_components]
    return eigenfaces


def project_faces(X_centered, eigenfaces):
    projections = np.dot(X_centered, eigenfaces.T)
    return projections


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size/1024:.2f} KB"
    else:
        return f"{size/(1024*1024):.2f} MB"


def show_image(image_path, title):
    img = cv2.imread(image_path)
    if img is None:
        print(f"\nGagal membaca gambar: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


print("\n==========================")
print("Deteksi Kemiripan Wajah")
print("Loading dataset...")

X, labels, filenames = load_dataset(
    "dataset" # folder dataset yang berisi subfolder untuk setiap orang dengan foto di dalamnya sebaigai data latih
)

print("Jumlah gambar :", len(X))

# Centering
X_centered, mean_face = center_data(X)

# PCA-SVD
eigenfaces = compute_pca_svd(X_centered, num_components=50)

# Proyeksi database
database_features = project_faces(X_centered, eigenfaces)

print("Training selesai.")
print("===========================")


def extract_feature(test_image_path, mean_face, eigenfaces):

    vector = preprocess_image(test_image_path)

    centered = vector - mean_face

    feature = np.dot(centered, eigenfaces.T)

    return feature


def recognize_euclidean(test_feature, database_features, labels):

    min_distance = float("inf")  # nilai awal untuk jarak minimum
    best_match = None

    for feature, label in zip(database_features, labels):

        dist = euclidean(test_feature, feature)

        if dist < min_distance:
            min_distance = dist
            best_match = label

    return best_match, min_distance


def recognize_cosine(test_feature, database_features, labels, filenames):

    best_score = -1
    best_match = None
    best_file = None

    test_feature = test_feature.reshape(1, -1)

    for feature, label, file in zip(database_features, labels, filenames):

        feature = feature.reshape(1, -1)

        score = cosine_similarity(test_feature, feature)[0][0]

        if score > best_score:
            best_score = score
            best_match = label
            best_file = file

    return best_match, best_file, best_score


for file in os.listdir(
    "test"# folder "test" yang berisi gambar yang akan diuji *"kompresi" bisa diganti dengan folder lain
):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        test_image = os.path.join("test", file)# folder "test" yang berisi gambar yang akan diuji *"kompresi" bisa diganti dengan folder lain

        print("\n================================")
        print("File Uji :", file)
        print("Ukuran File :", get_file_size(test_image))

        show_image(test_image, f"Gambar Uji\nUkuran File : {get_file_size(test_image)}")

        test_feature = extract_feature(test_image, mean_face, eigenfaces)

        top_results = recognize_cosine_topk(
            test_feature,
            database_features,
            labels,
            filenames,
            k=1,  # ubah k sesuai dengan yang dibutuhkan
        )

        print("\nTop Foto Yang Mirip :")

        for i, (label, foto, score) in enumerate(top_results, start=1):

            print(
                f"{i}. {foto} ({label}) "
                f"= {score:.4f} "
                f"= {get_file_size(os.path.join('dataset', label, foto))}"
            )

        best_label, best_file, best_score = top_results[0]

        best_path = os.path.join("dataset", best_label, best_file)

        print("\nHasil Terbaik")
        print("Nama :", best_file)
        print("Orang :", best_label)
        print("Similarity :", round(best_score, 4))
        print("Ukuran File :", get_file_size(best_path))

        show_image(
            best_path,
            f"Hasil Terbaik\nSimilarity : {best_score:.4f}\nUkuran File : {get_file_size(best_path)}",
        )
