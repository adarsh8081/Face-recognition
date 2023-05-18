# import PIL.Image
# import PIL.ImageDraw
# import face_recognition
#
# given_image = face_recognition.load_image_file('group.jpg')
# face_locations = face_recognition.face_locations(given_image)
# number_of_faces = len(face_locations)
# print("We found {} face(s) in this image. ".format(number_of_faces))
# pil_image = PIL.Image.fromarray(given_image)
#
# for face_location in face_locations :
#     top,left,bottom,right = face_location
#     print("A Face is detected at pixel location Top : {},Left {},Bottom {},Right {}".format(top,left,bottom,right))
#     draw = PIL.ImageDraw.Draw(pil_image)
#     draw.rectangle([left,top,right,bottom],outline="yellow",width=5)
#
#
# pil_image.show()
#
# import numpy as npy
#
#
# class K_means:
#     def _init_(factors, k=2, max_epochs=500):
#         factors.k = k
#         factors.max_epochs = max_epochs
#
#     def fit(factors, container):
#
#         container = container.values
#         factors.centroid = container[npy.random.randint(container.shape[0], size=factors.k), :]
#
#         for i in range(factors.max_epochs):
#             distances = npy.array([factors.euclidean(container, c) for c in factors.centroid])
#             factors.labels = npy.argmin(distances, axis=0)
#
#             for j in range(factors.k):
#                 points = container[factors.labels == j]
#                 factors.centroid[j] = npy.mean(points, axis=0)
#
#     def predict(factors, container):
#         container = container.values
#         distances = npy.array([factors.euclidean(container, c) for c in factors.centroid])
#         labels = npy.argmin(distances, axis=0)
#         return labels.astype(int)
#
#     def euclidean(factors, container, centroid):
#         distance = npy.sum((container - centroid) ** 2, axis=1)
#         return npy.sqrt(distance)

# Creating Function to measure Dunn Index

# import numpy as np
#
#
# def dunn_index(data, labels, k):
#     # Calculate minimum inter-cluster distance
#     min_inter_cluster_distance = float('inf')
#     for i in range(k):
#         for j in range(i + 1, k):
#             cluster_i = data.to_numpy()[labels == i]
#             cluster_j = data.to_numpy()[labels == j]
#             inter_cluster_distance = np.min(
#                 np.min(np.sqrt(np.sum((cluster_i - cluster_j[:, np.newaxis]) ** 2, axis=2)), axis=1))
#             min_inter_cluster_distance = min(min_inter_cluster_distance, inter_cluster_distance)
#
#     # Calculate maximum intra-cluster distance
#     max_intra_cluster_distance = 0
#     for i in range(k):
#         cluster_i = data.to_numpy()[labels == i]
#         intra_cluster_distance = np.max(np.sqrt(np.sum((cluster_i - cluster_i[:, np.newaxis]) ** 2, axis=2)))
#         max_intra_cluster_distance = max(max_intra_cluster_distance, intra_cluster_distance)
#
#     # Calculate Dunn Index
#     dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
#
#     return dunn_index
# 
# DI_lst = {"Clusters (k)":[], "DI (Dunn Index value)" : []}
# for i in range(len(labels_lst)):
#     k = i+2
#     label = labels_lst[i]
#     DI_lst["Clusters (k)"].append(k)
#     DI_lst["DI (Dunn Index value)"].append(dunn_index(df, label, k))
# 
# DI = pd.DataFrame.from_dict(DI_lst)
# DI.reset_index(drop=True, inplace=True)
# DI.head(10)

import numpy as np
# 
# class Fcm:
#     def __init__(factors, clusters, max_iter=300, m=2, epsiln=1e-5):
#         factors.clusters = clusters
#         factors.max_iter = max_iter
#         factors.m = m
#         factors.epsiln = epsiln
# 
#     def fit(factors, X):
#         X = X.to_numpy()
#         n_samples, n_features = X.shape
#         factors.mbrshp_mat = np.random.rand(n_samples, factors.clusters)
#         factors.mbrshp_mat /= factors.mbrshp_mat.sum(axis=1, keepdims=True)
#         factors.cluster_centers = np.random.rand(factors.clusters, n_features)
#         factors.prev_cluster_centers = np.zeros((factors.clusters, n_features))
# 
#         for i in range(factors.max_iter):
#             factors.cluster_centers = (X.T @ (factors.mbrshp_mat ** factors.m)) / (np.sum(factors.mbrshp_mat, axis=0) ** (factors.m - 1))
#             dist = np.linalg.norm(X[:, np.newaxis] - factors.cluster_centers, axis=2) ** 2
#             factors.mbrshp_mat = 1 / (dist / np.sum(dist, axis=1, keepdims=True) + 1)
#             if np.linalg.norm(factors.prev_cluster_centers - factors.cluster_centers) < factors.epsiln:
#                 break
#             factors.prev_cluster_centers = factors.cluster_centers.copy()
# 
#     def predict(factors, X):
#         X = X.to_numpy()
#         dist = np.linalg.norm(X[:, np.newaxis] - factors.cluster_centers, axis=2) ** 2
#         mbrshp_mat = 1 / (dist / np.sum(dist, axis=1, keepdims=True) + 1)
#         return np.argmax(mbrshp_mat, axis=1)
# 

