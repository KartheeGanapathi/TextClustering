import regex as re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
from collection import groupcount
from sk1earn import roundFunction
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.datasets import fetch_20newsgroups
news_data = fetch_20newsgroups(subset = 'all', random_state=1)

def dataInfo():
    st.header('Data Information')
    data = pd.DataFrame(news_data.data, columns=['count'])
    data['group'] = news_data.target
    data['group'] = data['group'].map(lambda i:news_data.target_names[i])

    groupCount = data.groupby('group').size().reset_index(name='og_count')
    st.write(groupCount)

    data = pd.DataFrame(news_data.data, columns=['count'])
    data['group'] = news_data.target
    data['group'] = data['group'].map(lambda i:news_data.target_names[i].split('.')[1])

    groupCount = data.groupby('group').size().reset_index(name='og_count')
    st.write(groupCount)

    data = pd.DataFrame(news_data.data, columns=['count'])
    data['group'] = news_data.target
    data['group'] = data['group'].map(lambda i:news_data.target_names[i].split('.')[0])

    groupCount = data.groupby('group').size().reset_index(name='og_count')
    st.write(groupCount)

def removeStopwords(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def lemmatization(text):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lWords = [lemmatizer.lemmatize(word) for word in words]
    lText = ' '.join(lWords)
    return lText

def preprocess(text):
    text = str(text.split(':')[-1])
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d_]+', '', text)
    text = removeStopwords(text)
    text = lemmatization(text)
    return text

textData = list(news_data.data)
for index, text in enumerate(textData):
    textData[index] = preprocess(text)

def tf_idf():
    st.header('TF - IDF')
    vectorizer = TfidfVectorizer()
    textVec = vectorizer.fit_transform(textData)

    st.subheader('PCA')
    textVecArray = textVec.toarray()
    pca = PCA(n_components=2)

    pca.fit(textVecArray)
    X_pca = pca.transform(textVecArray)

    st.subheader('K Means')
    nClusters = [7, 15, 20]
    silhouetteScore = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)
    for i in nClusters:
        kmeans = KMeans(n_clusters=i, random_state=69)
        kmeans.fit(X_pca)

        clusterLabels = kmeans.labels_
        silhouetteScore[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=75)
        st.pyplot(plt)
        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

    st.subheader('DBSCAN')
    eps, min_samples = 0.001, 26

    silhouette_scores = {}
    group_count = {}

    dbscan = DBSCAN(eps=0.001, min_samples=26)
    cluster_labels = dbscan.fit_predict(X_pca)

    silhouette_scores[(eps, min_samples)] = silhouette_score(X_pca, cluster_labels)

    cluster_count = pd.Series(cluster_labels).value_counts().sort_index()
    group_count[(eps, min_samples)] = cluster_count

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    st.pyplot(plt)
    st.write(f"EPS={eps}, Min Samples={min_samples}")
    st.write("Silhouette Score:", silhouetteScore[20])
    st.write("Cluster Counts:\n", groupcount[tuple((0.001, 26))])

    st.subheader('AGNES')
    nClusters = [7, 15, 20]
    silhouette_scorea = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)

    for i in nClusters:
        agnes = AgglomerativeClustering(n_clusters=i)
        clusterLabels = agnes.fit_predict(X_pca)

        silhouette_scores[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        dist_mat = pdist(X_pca)
        linkage_matrix = linkage(dist_mat, method='ward')

        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title(f'Dendrogram for {i} Clusters')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        st.pyplot(plt)

        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

def Word2Vec_():
    st.header('Word2Vec')
    tokenized_data = [word_tokenize(sentence.lower()) for sentence in textData]
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0
        for word in words:
            if word in vocabulary:
                n_words += 1
                feature_vector = np.add(feature_vector, model.wv[word])
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    word_vectors = []
    for words in tokenized_data:
        word_vectors.append(average_word_vectors(words, model, model.wv.index_to_key, 100))

    st.subheader('PCA')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(word_vectors)

    X_pca.shape

    st.subheader('K Means')
    nClusters = [7, 15, 20]
    silhouetteScore = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)
    for i in nClusters:
        kmeans = KMeans(n_clusters=i, random_state=69)
        kmeans.fit(X_pca)

        clusterLabels = kmeans.labels_
        silhouetteScore[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=75)
        st.pyplot(plt)
        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

    st.subheader('AGNES')
    nClusters = [7, 15, 20]
    silhouette_scores = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)

    for i in nClusters:
        agnes = AgglomerativeClustering(n_clusters=i)
        clusterLabels = agnes.fit_predict(X_pca)

        silhouette_scores[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        dist_mat = pdist(X_pca)
        linkage_matrix = linkage(dist_mat, method='ward')

        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title(f'Dendrogram for {i} Clusters')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        st.pyplot(plt)

        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

    st.subheader('DBSCAN')
    eps, min_samples = 0.0001, 15

    silhouette_scores = {}
    group_count = {}

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_pca)

    silhouette_scores[(eps, min_samples)] = silhouette_score(X_pca, cluster_labels)

    cluster_count = pd.Series(cluster_labels).value_counts().sort_index()
    group_count[(eps, min_samples)] = cluster_count

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    st.pyplot(plt)

    st.write(f"EPS={eps}, Min Samples={min_samples}")
    st.write("Silhouette Score:", silhouette_scores[(eps, min_samples)])
    st.write("Cluster Counts:\n", group_count[(eps, min_samples)])

def GloVE():
    st.header('GloVE')
    tokenized_data = [sentence.lower().split() for sentence in textData]
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=10, sg=1, negative=5, min_count=1, workers=4)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0
        for word in words:
            if word in vocabulary:
                n_words += 1
                feature_vector = np.add(feature_vector, model.wv[word])
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector

    word_vectors = []
    for words in tokenized_data:
        word_vectors.append(average_word_vectors(words, model, model.wv.index_to_key, 100))

    st.subheader('PCA')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(word_vectors)

    X_pca.shape

    st.subheader('K Means')
    nClusters = [7, 14, 20]
    silhouetteScore = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)
    for i in nClusters:
        kmeans = KMeans(n_clusters=i, random_state=69)
        kmeans.fit(X_pca)

        clusterLabels = kmeans.labels_
        silhouetteScore[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=75)
        st.pyplot(plt)
        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

    st.subheader('DBSCAN')
    eps, min_samples = 0.0001, 20

    silhouette_scores = {}
    group_count = {}

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_pca)

    silhouette_scores[(eps, min_samples)] = silhouette_score(X_pca, cluster_labels)

    cluster_count = pd.Series(cluster_labels).value_counts().sort_index()
    group_count[(eps, min_samples)] = cluster_count

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=20, alpha=0.7, edgecolors='w')
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    st.pyplot(plt)

    st.write(f"EPS={eps}, Min Samples={min_samples}")
    st.write("Silhouette Score:", silhouette_scores[(eps, min_samples)])
    st.write("Cluster Counts:\n", group_count[(eps, min_samples)])

    st.subheader('AGNES')
    nClusters = [7, 15, 20]
    silhouette_scorea = defaultdict(float)
    groupCount = defaultdict(pd.DataFrame)

    for i in nClusters:
        agnes = AgglomerativeClustering(n_clusters=i)
        clusterLabels = agnes.fit_predict(X_pca)

        silhouette_scores[i] = silhouette_score(X_pca, clusterLabels)

        clusterCount = np.bincount(clusterLabels)
        tempDict = {}
        for index, count in enumerate(clusterCount):
            tempDict[index] = count
        tempDict = pd.DataFrame.from_dict(tempDict, orient='index', columns=['count'])
        groupCount[i] = tempDict

        dist_mat = pdist(X_pca)
        linkage_matrix = linkage(dist_mat, method='ward')

        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title(f'Dendrogram for {i} Clusters')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        st.pyplot(plt)

        st.write(i, silhouetteScore[i])
        st.write(groupcount[i])

def about_us():
    st.title("THANK YOU")
    st.subheader("About us")
    st.write("done by : ")
    st.write("Aditya Ramanathan 20PD02")
    st.write("Kartheepan G      20pd11")


page = {
"TF - IDF" : tf_idf,
"Word2Vec" : Word2Vec_,
"GloVE" : GloVE,
"Data" : dataInfo,
"About us" : about_us
}

pages = st.sidebar.selectbox("select the page :", page.keys())
page[pages]()