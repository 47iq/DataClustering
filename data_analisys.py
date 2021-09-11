import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def parse_csv_data(file):
    mall_customers_data = pd.read_csv(
        file,
        delimiter=',',
        names=['id', 'gender', 'age', 'salary', 'score']
    )
    X = mall_customers_data['salary']
    y = mall_customers_data['score']
    xy_data = np.vstack((X, y)).T
    return xy_data
