from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from sklearn.decomposition import PCA # Wassim
from sklearn.cluster import KMeans # Wassim
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score # Wassim
from sklearn.manifold import TSNE # library used for tsne - Mohamad
from sklearn.cluster import KMeans # library used for clustering  Mohamad
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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

def dim_red_lda(mat, p):  
    '''
    Perform dimensionality reduction using Linear Discriminant Analysis (LDA)

    Input:
    -----
        mat : NxM list or array-like 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP array such that p<<m
    '''
    lda = LDA(n_components=p)
    red_mat = lda.fit_transform(mat, labels)  # Assuming 'labels' is defined in your code
    return red_mat



def dim_red_nmf(mat, p):  # Exemple d'une autre méthode de réduction de dimension
    '''
    Perform dimensionality reduction using NMF (Non-negative Matrix Factorization)

    Input:
    -----
        mat : NxM list or array-like 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP array such that p<<m
    '''
    # Assurer que les données sont non négatives en les normalisant
    scaler = MinMaxScaler()
    mat_non_negative = scaler.fit_transform(mat)

    nmf = NMF(n_components=p)
    red_mat = nmf.fit_transform(mat_non_negative)
  
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

# # Charger les données
# ng20 = fetch_20newsgroups(subset='test')
# corpus = ng20.data[:2000]
# labels = ng20.target[:2000]

# # Embeddings
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# embeddings = model.encode(corpus)

with open('ng20_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)


# Initialiser la variable models_to_test avec une liste vide
models_to_test = []

# Demander à l'utilisateur de saisir les modèles à tester jusqu'à ce qu'une valeur valide soit fournie
while not models_to_test:
    models_to_test = input("Enter model to test (example: UMAP OR ACP OR tsne): ").split(',')

    # Vérifier si les modèles saisis sont valides
    for model_name in models_to_test:
        if model_name.strip().lower() not in ['umap', 'tsne', 'acp']:
            print(f"Invalid model '{model_name}'. Please enter valid models.")
            models_to_test = []  # Réinitialiser la liste pour relancer la boucle
            break

# Demander à l'utilisateur de saisir le nombre de folds pour la validation croisée
num_folds = int(input("Enter the number of folds for cross-validation: "))

# Utiliser KFold pour obtenir les indices d'entraînement et de test
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# ACP - Wassim
# perform dimentionality reduction
red_emb = dim_red_acp(embeddings, 20)

# Créer une unique figure avec une grille plus grande
fig, axs = plt.subplots(len(models_to_test), num_folds * 3, figsize=(15 * num_folds, 5 * len(models_to_test)))

# Boucle sur chaque modèle
for i, model_name in enumerate(models_to_test):
    total_nmi = 0.0
    total_ari = 0.0

    # Boucle sur chaque fold
    for fold, (train_index, test_index) in enumerate(kf.split(embeddings)):
        index = fold * 3  # Utiliser une variable temporaire pour simplifier l'indexation

        # Séparer les données en ensembles d'apprentissage et de test
        train_embeddings, test_embeddings = embeddings[train_index], embeddings[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Clustering
        if model_name.strip().lower() == 'umap':
            red_emb = dim_red_UMAP(train_embeddings, 2, method = 'UMAP')
        elif model_name.strip().lower() == 'tsne':
            red_emb = dim_red_tsne(train_embeddings, 2)
        elif model_name.strip().lower() == 'acp':
            red_emb = dim_red_acp(train_embeddings, 2)
        else:
            print(f"Model '{model_name}' not recognized. Skipping.")
            continue

        pred = clust(red_emb, k)
        nmi_score = normalized_mutual_info_score(pred, train_labels)
        ari_score = adjusted_rand_score(pred, train_labels)

        # Accumuler les scores NMI et ARI
        total_nmi += nmi_score
        total_ari += ari_score

        scatter = axs[index].scatter(red_emb[:, 0], red_emb[:, 1], c=pred, cmap='viridis', alpha=0.7, s=30)
        axs[index].set_title(f'{model_name.upper()} Visualization (Fold {fold + 1})')
        axs[index].legend(*scatter.legend_elements(), title="Clusters")

        axs[index + 1].text(0.5, 0.5, f'NMI: {nmi_score:.2f}\nARI: {ari_score:.2f}', fontsize=12, va='center')
        axs[index + 1].axis('off')

        axs[index + 2].axis('off')

    # Calculer les moyennes des scores NMI et ARI sur tous les folds
    avg_nmi = total_nmi / num_folds
    avg_ari = total_ari / num_folds

    # Afficher les moyennes dans le sous-graphique correspondant
    axs[num_folds * 3 - 1].text(0.5, 0.5, f'Average NMI: {avg_nmi:.2f}\nAverage ARI: {avg_ari:.2f}', fontsize=12, va='center')
    axs[num_folds * 3 - 1].axis('off')

# Imprimer le nombre de modèles après la boucle
print("Nombre total de modèles :", len(models_to_test))

plt.tight_layout()
plt.show()

print(f'By UMAP Method : NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')
