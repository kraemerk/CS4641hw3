#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import fowlkes_mallows_score,homogeneity_completeness_v_measure, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale, OrdinalEncoder, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from time import time



# Silences sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# Run kmeans on input X and y and evaluate and plot
def kmeans(X,y, dataset_name):

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=65)
    train_scores = []
    train_homo = []
    train_completeness = []
    train_v_score = []

    test_scores = []
    test_homo = []
    test_completeness = []
    test_v_score = []

    kvals = [x for x in range(2,51)]

    for k in range(2, 51):
        print("k= {}".format(k))
        clf = KMeans(n_clusters=k, max_iter=1000)
        # Train on train data, recording accuracy, homogeneity, completeness, and v_measure
        train_pred = clf.fit_predict(X_train)
        train_score = fowlkes_mallows_score(y_train, train_pred)
        train_scores.append(train_score)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_train, train_pred)
        train_homo.append(homogeneity)
        train_completeness.append(completeness)
        train_v_score.append(v_measure)

        # Evaluate same metrics on test set
        test_pred = clf.predict(X_test)
        test_score = fowlkes_mallows_score(y_test, test_pred)
        test_scores.append(test_score)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_test, test_pred)
        test_homo.append(homogeneity)
        test_completeness.append(completeness)
        test_v_score.append(v_measure)

    print("done")
    print('best performing number of clusters: {}'.format(kvals[np.argmax(test_v_score)]))
    print("generating plots")

    plt.figure()
    plt.title('Folkes-Mallows Score of K-Means on {} Dataset'.format(dataset_name))
    plt.xlabel('K Value (Number of Clusters)')
    plt.ylabel('Folkes-Mallows Score')
    plt.plot(kvals, train_scores, label='Training Score')
    plt.plot(kvals, test_scores, label='Test Score')
    plt.legend(loc='upper left')
    plt.show(block=False)

    plt.figure()
    plt.title('Performance Metrics of K-Means on {} Dataset'.format(dataset_name))
    plt.xlabel('K Value (Number of Clusters)')
    plt.ylabel('Score (Range 0.0 to 1.0)')
    plt.plot(kvals, train_homo, label='Training Homogeneity')
    plt.plot(kvals, test_homo, label='Test Homogeneity')
    plt.plot(kvals, train_completeness, label='Training Completeness')
    plt.plot(kvals, test_completeness, label='Test Completeness')
    plt.plot(kvals, train_v_score, label='Training V-Measure')
    plt.plot(kvals, test_v_score, label='Test V-Measure')
    plt.legend(loc='upper left')
    plt.show(block=False)

def expectation_maximization(X,y,dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=65)
    train_scores = []
    train_homo = []
    train_completeness = []
    train_v_score = []

    test_scores = []
    test_homo = []
    test_completeness = []
    test_v_score = []

    kvals = [x for x in range(2,51)]

    for k in range(2, 51):
        print("k= {}".format(k))
        clf = GaussianMixture(n_components=k, max_iter=1000)
        # Train on train data, recording accuracy, homogeneity, completeness, and v_measure
        train_pred = clf.fit_predict(X_train)
        train_score = fowlkes_mallows_score(y_train, train_pred)
        train_scores.append(train_score)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_train, train_pred)
        train_homo.append(homogeneity)
        train_completeness.append(completeness)
        train_v_score.append(v_measure)

        # Evaluate same metrics on test set
        test_pred = clf.predict(X_test)
        test_score = fowlkes_mallows_score(y_test, test_pred)
        test_scores.append(test_score)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_test, test_pred)
        test_homo.append(homogeneity)
        test_completeness.append(completeness)
        test_v_score.append(v_measure)

    print("done")
    print("generating plots")

    plt.figure()
    plt.title('Folkes-Mallows Score of Expectation Maximization on {} Dataset'.format(dataset_name))
    plt.xlabel('Number of Components')
    plt.ylabel('Folkes-Mallows Score')
    plt.plot(kvals, train_scores, label='Training Score')
    plt.plot(kvals, test_scores, label='Test Score')
    plt.legend(loc='upper left')
    plt.show(block=False)

    plt.figure()
    plt.title('Performance Metrics of Expectation Maximization on {} Dataset'.format(dataset_name))
    plt.xlabel('K Value (Number of Clusters)')
    plt.ylabel('Score (Range 0.0 to 1.0)')
    plt.plot(kvals, train_homo, label='Training Homogeneity')
    plt.plot(kvals, test_homo, label='Test Homogeneity')
    plt.plot(kvals, train_completeness, label='Training Completeness')
    plt.plot(kvals, test_completeness, label='Test Completeness')
    plt.plot(kvals, train_v_score, label='Training V-Measure')
    plt.plot(kvals, test_v_score, label='Test V-Measure')
    plt.legend(loc='upper left')
    plt.show(block=False)


# Shows a visualization of the input data as a 3D scatterplot. 
def visualize_data_3D(X,y, dataset_name):
    #Show the input data projected into 3D
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    plt.cla()
    plt.title('Original data projected into 3 Dimensions: {} Data'.format(dataset_name))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show(block=False)

# Shows a visualization of the input data as a 2D scatterplot. 
def visualize_data_2D(X,y, dataset_name):
    plt.figure()
    plt.title('Original data projected into 2 Dimensions: {} Data'.format(dataset_name))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show(block=False)


def pca(X,y, dataset_name):
    # Transform into 2 componenets first
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    plt.figure()
    plt.title('{} data after PCA decomposition into 2 components'.format(dataset_name))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    plt.show(block=False)

    # Transform into 3 components
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    # Visualize transformed data
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    plt.cla()
    plt.title('{} data after PCA decomposition into 3 components'.format(dataset_name))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show(block=False)

def ica(X,y, dataset_name):
    # Transform into 2 components first
    ica = FastICA(n_components=2)

    X_transformed = ica.fit_transform(X)
    plt.figure()
    plt.title('{} data after ICA decomposition into 2 components'.format(dataset_name))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    plt.show(block=False)

    ica = FastICA(n_components=3)
    X_transformed = ica.fit_transform(X)
    
    # Visualize transformed data
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    plt.cla()
    plt.title('{} data after ICA decomposition into 3 components'.format(dataset_name))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show(block=False)

def randomized_projection(X,y, dataset_name):
    rand = GaussianRandomProjection(n_components=2)

    X_transformed = rand.fit_transform(X)
    plt.figure()
    plt.title('{} data after Gaussian Random Projection into 2 components'.format(dataset_name))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    plt.show(block=False)

    rand = GaussianRandomProjection(n_components=3)
    X_transformed = rand.fit_transform(X)
    # Visualize transformed data
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    plt.cla()
    plt.title('{} data after Gaussian Random Projection into 3 components'.format(dataset_name))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show(block=False)

def select_k_best(X,y, dataset_name):
    select = SelectKBest(f_classif, k=2)

    X_transformed = select.fit_transform(X,y)
    plt.figure()
    plt.title('{} data after 2 best features selected'.format(dataset_name))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    plt.show(block=False)

    select = SelectKBest(f_classif, k=3)
    X_transformed = select.fit_transform(X,y)
    
    # Visualize transformed data
    plt.figure()
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
    plt.cla()
    plt.title('{} data after 3 best features selected'.format(dataset_name))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show(block=False)


def reduce_then_cluster(X,y, dataset_name):
    # First, PCA n=2
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    kmeans(X_transformed, y, dataset_name + ' - After PCA (n_components=2)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After PCA (n_components=2)')

    # Then, PCA n=3
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    kmeans(X_transformed, y, dataset_name + ' - After PCA (n_components=3)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After PCA (n_components=3)')

    # ICA, n=2
    ica = FastICA(n_components=2)
    X_transformed = ica.fit_transform(X)

    kmeans(X_transformed, y, dataset_name + ' - After ICA (n_components=2)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After ICA (n_components=2)')

    # ICA, n=3
    ica = FastICA(n_components=2)
    X_transformed = ica.fit_transform(X)

    kmeans(X_transformed, y, dataset_name + ' - After ICA (n_components=3)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After ICA (n_components=3)')

    # Random Projections, n=2
    rand = GaussianRandomProjection(n_components=2, random_state=65)
    X_transformed = rand.fit_transform(X)
    kmeans(X_transformed, y, dataset_name + ' - After Gaussian Random Projection (n_components=2)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After Gaussian Random Projection (n_components=2)')

    # Random Projections, n=3
    rand = GaussianRandomProjection(n_components=3, random_state=65)
    X_transformed = rand.fit_transform(X)
    kmeans(X_transformed, y, dataset_name + ' - After Gaussian Random Projection (n_components=3)')
    expectation_maximization(X_transformed, y, dataset_name + ' - After Gaussian Random Projection (n_components=3)')

    # Select K best, k=2
    select = SelectKBest(f_classif, k=2)
    X_transformed = select.fit_transform(X,y)
    kmeans(X_transformed, y, dataset_name + ' - After 2 Best Features Selected')
    expectation_maximization(X_transformed, y, dataset_name + ' - After 2 Best Features Selected')

    # Select K best, k=3
    select = SelectKBest(f_classif, k=3)
    X_transformed = select.fit_transform(X,y)
    kmeans(X_transformed, y, dataset_name + ' - After 3 Best Features Selected')
    expectation_maximization(X_transformed, y, dataset_name + ' - After 3 Best Features Selected')

def run_neural_net(X,y, finalX, finalY):
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    # Get learning curve data 
    start = time()
    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.array([0.55, 0.65, 0.75, 0.85, 0.95]))
    print("cross validation time: {}".format(time() - start))

    # Finally, train on cross validation data and test on withheld data
    clf.fit(X, y)
    y_pred = clf.predict(finalX)

    return train_sizes, train_scores, valid_scores, finalY, y_pred




def reduce_then_neural_net(X,y, dataset_name, max_components):
    # Baseline
    print('BASELINE TEST FOR {} DATA'.format(dataset_name))
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=65)
    _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
    print('mean training score: {}'.format(np.mean(train_scores)))
    print('mean validation score: {}'.format(np.mean(valid_scores)))
    print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))

    pca_training_means = []
    pca_validation_means = []
    pca_test_scores = []

    ica_training_means = []
    ica_validation_means = []
    ica_test_scores = []

    rand_training_means = []
    rand_validation_means = []
    rand_test_scores = []

    selectk_training_means = []
    selectk_validation_means = []
    selectk_test_scores = []

    kvals = [k for k in range(2, max_components + 1)]

    for k in range(2, max_components + 1):
        pca = PCA(n_components=k)
        ica = FastICA(n_components=k)
        rand = GaussianRandomProjection(n_components=k, random_state=65)
        select = SelectKBest(f_classif, k=k)
        print()
        print('TESTS WITH NUM_COMPONENTS={} for {} data'.format(k,dataset_name))
        print('PCA NEURAL NET STATS')
        X_pca = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_pca,y, random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        pca_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        pca_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        pca_test_scores.append(accuracy_score(y_test, y_pred))

        print()
        print('ICA NEURAL NET STATS')
        X_ica = ica.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_ica,y, random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        ica_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        ica_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        ica_test_scores.append(accuracy_score(y_test, y_pred))

        print()
        print('RANDOMIZED PROJECTIONS NEURAL NET STATS')
        X_rand = rand.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_rand,y, random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        rand_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        rand_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        rand_test_scores.append(accuracy_score(y_test, y_pred))

        print()
        print('SELECT-K NEURAL NET STATS')
        X_select = select.fit_transform(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X_select,y, random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        selectk_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        selectk_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        selectk_test_scores.append(accuracy_score(y_test, y_pred))

    plt.figure()
    plt.title('Neural Net Results After Decomposition - {} Data'.format(dataset_name))
    plt.plot(kvals, pca_training_means, label='PCA Mean Training Score')
    plt.plot(kvals, pca_validation_means, label='PCA Mean Validation Score')
    plt.plot(kvals, pca_test_scores, label='PCA Test Score')
    plt.plot(kvals, ica_training_means, label='ICA Mean Training Score')
    plt.plot(kvals, ica_validation_means, label='ICA Mean Validation Score')
    plt.plot(kvals, ica_test_scores, label='ICA Test Score')
    plt.plot(kvals, rand_training_means, label='Randomized Projection Training Score')
    plt.plot(kvals, rand_validation_means, label='Randomized Projection Validation Score')
    plt.plot(kvals, rand_test_scores, label='Randomized Projection Test Score')
    plt.plot(kvals, selectk_training_means, label='Select K Best Training Score')
    plt.plot(kvals, selectk_validation_means, label='Select K Best Validation Score')
    plt.plot(kvals, selectk_test_scores, label='Select K Best Test Score')
    plt.legend(loc='upper left')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.show(block=False)

def cluster_then_neural_net(X,y, dataset_name):
    kmeans_training_means = []
    kmeans_validation_means = []
    kmeans_test_scores = []

    expectation_maximization_training_means = []
    expectation_maximization_validation_means = []
    expectation_maximization_test_scores = []

    kvals = [k for k in range(2,21)]

    # Baseline
    print('BASELINE TEST FOR {} DATA'.format(dataset_name))
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=65)
    _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
    print('mean training score: {}'.format(np.mean(train_scores)))
    print('mean validation score: {}'.format(np.mean(valid_scores)))
    print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))

    for num_clusters in range(2, 21):
        kmeans = KMeans(n_clusters=num_clusters)    
        expectation_maximization = GaussianMixture(n_components=num_clusters)
        print('TESTS WITH NUM_CLUSTERS={} for {} data'.format(num_clusters,dataset_name))
        # First, kmeans
        print('K-MEANS NEURAL NET STATS')
        X_kmeans = kmeans.fit_predict(X).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_kmeans,y,random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        kmeans_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        kmeans_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        kmeans_test_scores.append(accuracy_score(y_test, y_pred))

        # Then EM
        print()
        print('EXPECTATION MAXIMIZATION NEURAL NET STATS')
        X_em = expectation_maximization.fit_predict(X).reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X_em,y,random_state=65)
        _, train_scores, valid_scores, finalY, y_pred = run_neural_net(X_train, y_train, X_test, y_test)
        print('mean training score: {}'.format(np.mean(train_scores)))
        expectation_maximization_training_means.append(np.mean(train_scores))
        print('mean validation score: {}'.format(np.mean(valid_scores)))
        expectation_maximization_validation_means.append(np.mean(valid_scores))
        print('Test accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        expectation_maximization_test_scores.append(accuracy_score(y_test, y_pred))

    plt.figure()

    plt.title('Neural Net Results after Clustering Algorithms - {} Data'.format(dataset_name))
    plt.plot(kvals, kmeans_training_means, label='K-Means Mean Training Score')
    plt.plot(kvals, kmeans_validation_means, label='K-Means Mean Validation Score')
    plt.plot(kvals, kmeans_test_scores, label='K-Means Test Score')
    plt.plot(kvals, expectation_maximization_training_means, label='Expectation Maximization Mean Training Score')
    plt.plot(kvals, expectation_maximization_validation_means, label='Expectation Maximization Mean Validation Score')
    plt.plot(kvals, expectation_maximization_test_scores, label='Expectation Maximization Test Score')
    plt.legend(loc='upper left')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.show(block=False)

def load_data():
    data = pd.read_csv('car-data.csv')

    X,y = data.values[:, 0:6], data.values[:, 6]
    enc = OrdinalEncoder()
    enc.fit(X)
    X = enc.transform(X)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return X,y

def main():
    X,y = load_data()
    visualize_data_2D(X,y, 'Car')
    visualize_data_3D(X,y, 'Car')
    kmeans(X,y, 'Car')
    expectation_maximization(X,y, 'Car')
    pca(X,y,'Car')
    ica(X,y, 'Car')
    randomized_projection(X,y, 'Car')
    select_k_best(X,y, 'Car')
    reduce_then_cluster(X,y, 'Car')
    plt.show()

if __name__ == "__main__":
    main()

