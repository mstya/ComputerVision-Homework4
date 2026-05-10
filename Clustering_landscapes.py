import os
import shutil
from pathlib import Path

import cv2
import keras.utils as image
import numpy as np
from keras import Model
from keras.src.applications.vgg16 import VGG16
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.cluster import KMeans
from keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def normalize_flatten(images):
    return [image
            .flatten()
            for image in images]

def normalise_with_hist(images):
    normalised_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        normalised_images.append(hist.flatten())
    return normalised_images

def extract_features(file, model):
    img = image.load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx)
    return features

def extract_features_with_vgg16(data):
    featuresList = []
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    for image in data:
        imgx = preprocess_input(image)
        features = model.predict(imgx)
        featuresList.append(features)
    return np.array(featuresList).reshape(-1, 4096)

def read_data(source_folder: str, categories: list[str], examples_per_category_count: int):
    data = []
    for ci, category in enumerate(categories):
        full_path = os.path.join(source_folder, category)
        with os.scandir(full_path) as files:
            items = list(files)
            for i in range(examples_per_category_count + 1):
                if items[i].name.endswith('.jpg'):
                    img = image.load_img(items[i].path, target_size=(224, 224))
                    img = np.array(img)
                    img = img.reshape(1, 224, 224, 3)
                    data.append((ci, img, items[i].path))

    # завантаження всього дата-сету (дуже довго буде працювати)
    # dataset_path = "./Aerial_Landscapes"
    # for ci, class_name in enumerate(os.listdir(dataset_path)):
    #     class_dir = os.path.join(dataset_path, class_name)
    #     if not os.path.isdir(class_dir):
    #         continue
    #     for fname in os.listdir(class_dir):
    #         if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
    #             path = os.path.join(class_dir, fname)
    #             img = image.load_img(path, target_size=(224, 224))
    #             img = np.array(img)
    #             img = img.reshape(1, 224, 224, 3)
    #             data.append((ci, img, path))

    return data

def view_cluster(groups, cluster):
    plt.figure(figsize=(25, 25))
    files = groups[cluster]
    if len(files) > 100:
        print(f"Clipping cluster size from {len(files)} to 100")
        files = files[:100]

    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = image.load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

def recreateTargetDir(dirToDelete: str) -> None:
    try:
        if Path("my_folder").is_dir():
            shutil.rmtree(dirToDelete)
        os.makedirs(dirToDelete)
    except OSError as e:
        print(e)
        pass

def show_confusion_matrix(true_labels, predictions, k):
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(12, 10))
    heatmap(cm,
            xticklabels=[f"Cluster {i}" for i in range(k)],
            yticklabels=categories,
            annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Передбачений кластер")
    plt.ylabel("Справжній клас")
    plt.title("Confusion Matrix: K-means кластеризація")
    plt.tight_layout()
    plt.show()

def show_elbow(features, K_range):
    wcss = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, 'bo-')
    plt.xlabel('Кількість кластерів K')
    plt.ylabel('WCSS (інерція)')
    plt.title('Elbow Method - вибір оптимального K')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()

def show_silhouette(features, K_range):
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'ro-')
    plt.xlabel('Кількість кластерів K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score - вибір оптимального K')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()

def applyClustering(source_folder: str, categories: list[str], examples_per_category_count: int, k: int) -> None:
    data = read_data(source_folder, categories, examples_per_category_count)
    image_paths = [image[2] for image in data]
    images = [image[1] for image in data]
    true_labels = [image[0] for image in data]
    # normalized_images = normalize_flatten(images)
    # normalized_images = normalise_with_hist(images)
    normalized_images = extract_features_with_vgg16(images)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(normalized_images)

    pca = PCA(n_components=50, random_state=0)
    features_reduced = pca.fit_transform(features_scaled)

    kmeans = (KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
              .fit(features_reduced))
    sil = silhouette_score(normalized_images, kmeans.labels_)
    print(f"Silhouette score: {sil}")

    targetdir = "./Target_Landscapes/"
    recreateTargetDir(targetdir)

    for i, m in enumerate(kmeans.labels_):
        print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
        shutil.copy(image_paths[i], targetdir + str(m) + "_" + str(i) + ".jpg")

    groups = {}
    for file, cluster in zip(image_paths, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    for i in range(k):
        view_cluster(groups, i)
    plt.show()

    ari = adjusted_rand_score(true_labels, kmeans.labels_)
    nmi = normalized_mutual_info_score(true_labels, kmeans.labels_)
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}")

    K_range = range(2, 20)
    show_confusion_matrix(true_labels, kmeans.labels_, k)
    show_elbow(features_reduced, K_range)
    show_silhouette(features_reduced, K_range)

# Замість Vgg16 спробував заради цікавості просто вектора чисел, без фіч, правює очевидно погано.
# Спробував також із гістограмою кольорів, теж погано, бо наприклад поле від лісу відрізнити воно не може.
# Із Vgg16 кластерізує майже ідеально.
#
# Щодо метрик.
# Створив логіку яка запускає кластерізацію для для K від 2 до 20.
# Якщо дивитись методом ліктя, то K повинну дорівнювати 4, що вірно.
# Якщо дивитись на силует, то найбільше значення при K=5.

# Подивився на додаткові метріки, вони корисні коли відома кількість K.
# Adjusted Rand Index - попарно дивиться на зображення і рахує кількість пар які згрупповані правильно. 0 = випадково, 1 = ідеально
# ARI = 0.9932
# Normalized Mutual Info - оцінює наскільки кластери несуть інформацію про класи
# NMI = 0.9898
# Ще додатково можна побудувати так звану confusion matrix. Вона показує скільки прикладів із конкретного тренувального набору потрапило до якого кластеру.

# З цікавого, спочатку я по кожній категорії використовував по 10 зображень, і результат був набагато гірше
# ARI: 0.5818
# NMI: 0.7031
# Зараз при 100 зображень для кожної категорії результати майже ідеальні.

if __name__ == "__main__":
    source_folder = "./Aerial_Landscapes"
    categories = [
        "Agriculture",
        "City",
        "Desert",
        "Forest"
    ]
    # examples_per_category_count = 10
    examples_per_category_count = 100

    applyClustering(source_folder, categories, examples_per_category_count, 4)
