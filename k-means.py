import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import data_analisys as da


def perform_kmeans(data, n_clusters):
    kmeansModel = KMeans(n_clusters=n_clusters, random_state=0)
    kmeansModel.fit(data)
    labels = kmeansModel.labels_
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()


def print_clusters_num_plot(data):
    criteria = []
    for k in range(2, 15):
        kmeansModel = KMeans(n_clusters=k, random_state=3)
        kmeansModel.fit(data)
        criteria.append(kmeansModel.inertia_)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.plot(range(2, 15), criteria)
    plt.show()


xy_data = da.parse_csv_data('Mall_Customers.csv')
# print_clusters_num_plot(xy_data)
perform_kmeans(xy_data, 10)
