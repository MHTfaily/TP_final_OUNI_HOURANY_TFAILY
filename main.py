from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # Wassim
from sklearn.decomposition import PCA # Wassim
from sklearn.cluster import KMeans # Wassim
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score # Wassim
from sklearn.manifold import TSNE # library used for tsne - Mohamad
from sklearn.cluster import KMeans # library used for clustering  Mohamad
# from umap import UMAP


def dim_red_UMAP(mat, p, method): #Joe
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method == 'umap':
        umap_model = UMAP(n_components=p)
        red_mat = umap_model.fit_transform(mat)
    else:
        # Add other dimensionality reduction methods here if needed
        red_mat = mat[:, :p]
        
    return red_mat

def dim_red_tsne(mat, p):  # Mohamad
    '''
    Perform dimensionality reduction using t-SNE

    Input:
    -----
        mat : NxM list or array-like 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP array such that p<<m
    '''
    # Convert the input list to a NumPy array if it's not already an array
    mat_np = mat if isinstance(mat, np.ndarray) else np.array(mat)
    
    # Initialize t-SNE model with desired number of components
    tsne = TSNE(n_components=p)
    
    # Fit and transform the data to the lower-dimensional space
    red_mat = tsne.fit_transform(mat_np)
    
    return red_mat

def dim_red_acp(mat, p):  # Wassim
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    
    
    pca = PCA(n_components=p)
    
    # Appliquer l'ACP et réduire la dimensionnalité
    red_mat = pca.fit_transform(mat)
    
    
    
   # red_mat = mat[:,:p]
    
    return red_mat


def clust(mat, k):
    '''
    Perform clustering using KMeans

    Input:
    -----
        mat : input list or array-like
        k : number of clusters
    Output:
    ------
        pred : list of predicted labels
    '''
    # Convert the input list to a NumPy array if it's not already an array
    mat_np = mat if isinstance(mat, np.ndarray) else np.array(mat)
    
    # Initialize KMeans model with the desired number of clusters
    kmeans = KMeans(n_clusters=k)
    
    # Fit KMeans to the data and predict cluster labels
    pred = kmeans.fit_predict(mat_np)
    
    return pred.tolist()


# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# TSNE - Mohamad
# perform dimentionality reduction
red_emb = dim_red_tsne(embeddings, 3)

# perform clustering
pred = clust(red_emb, k)

# evaluate clustering results
nmi_score = normalized_mutual_info_score(pred,labels)
ari_score = adjusted_rand_score(pred,labels)

print(f'By TSNE method: NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')


# ACP - Wassim
# perform dimentionality reduction
red_emb = dim_red_acp(embeddings, 20)

# perform clustering
pred = clust(red_emb, k)

# evaluate clustering results
nmi_score = normalized_mutual_info_score(pred,labels)
ari_score = adjusted_rand_score(pred,labels)

print(f'By ACP method: NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')


#UMAP Joe
# perform dimentionality reduction
red_emb = dim_red_UMAP(embeddings, 20, method = 'UMAP')

# perform clustering
pred = clust(red_emb, k)

# evaluate clustering results
nmi_score = normalized_mutual_info_score(pred,labels)
ari_score = adjusted_rand_score(pred,labels)

print(f'By UMAP Method : NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')

