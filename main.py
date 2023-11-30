from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from sklearn.decomposition import PCA # Wassim
from sklearn.cluster import KMeans # Wassim
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score # Wassim
from sklearn.manifold import TSNE # library used for tsne - Mohamad
from sklearn.cluster import KMeans # library used for clustering  Mohamad
import pickle

def dim_red_UMAP(mat, p, method): #Joe
    if method == 'umap':
        umap_model = UMAP(n_components=p)
        red_mat = umap_model.fit_transform(mat)
    else:
        # Add other dimensionality reduction methods here if needed
        red_mat = mat[:, :p]
        
    return red_mat

def dim_red_tsne(mat, p):  # Mohamad

    # Convert the input list to a NumPy array if it's not already an array
    mat_np = mat if isinstance(mat, np.ndarray) else np.array(mat)
    
    # Initialize t-SNE model with desired number of components
    tsne = TSNE(n_components=p)
    
    # Fit and transform the data to the lower-dimensional space
    red_mat = tsne.fit_transform(mat_np)
    
    return red_mat

def dim_red_acp(mat, p):  # Wassim      
    pca = PCA(n_components=p)
    # Appliquer l'ACP et réduire la dimensionnalité
    red_mat = pca.fit_transform(mat)
   # red_mat = mat[:,:p]
    
    return red_mat


def clust(mat, k):
    # Convert the input list to a NumPy array if it's not already an array
    mat_np = mat if isinstance(mat, np.ndarray) else np.array(mat)
    
    # Initialize KMeans model with the desired number of clusters
    kmeans = KMeans(n_clusters=k)
    
    # Fit KMeans to the data and predict cluster labels
    pred = kmeans.fit_predict(mat_np)
    
    return pred.tolist()

with open('ng20_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

k = len(set(labels))

with open('ng20_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

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