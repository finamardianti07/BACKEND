import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    svm.SVC(kernel='rbf')
)

# path dataset
data_dir = "dataset/train"

categories = ["bercak", "Hawar", "Healthy"]

data = []
labels = []

# load gambar
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)

        if img is None:
            print("Gagal Baca: ", img_path)
            continue
        img = cv2.resize(img, (64, 64))  # resize biar konsisten
        img = img.flatten()

        data.append(img)
        labels.append(label)

# cek data
print("Jumlah Data: ", len(data))
if len(data) == 0:
    print("Dataset Kosong / Tidak Terbaca")
    exit()

# ubah ke numpy array
data = np.array(data)
labels = np.array(labels)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size= 0.2, random_state=42
)

# buat model SVM
model = svm.SVC(kernel='rbf', probability=True)
# training
model.fit(X_train, y_train)
# evaluasi
y_pred = model.predict(X_test)
print("Akurasi: ", accuracy_score(y_test, y_pred))

# ✅ simpan model
joblib.dump(model, "model_svm.pkl")

print("✅ Model berhasil disimpan sebagai model_svm.pkl")
