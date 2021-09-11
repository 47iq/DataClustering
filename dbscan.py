from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import data_analisys as da


def perform_dbscan(data):
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)
    clustering = DBSCAN(eps=10, min_samples=5).fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=clustering)
    plt.show()


xy_data = da.parse_csv_data('Mall_Customers.csv')
# da.print_clusters_num_plot(xy_data)
perform_dbscan(xy_data)
