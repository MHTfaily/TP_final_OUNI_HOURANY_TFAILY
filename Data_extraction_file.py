# We just used this code to download the data and create the files ng20_embeddings.pkl and ng20_labels.pkl

from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import pickle

# Fetch the dataset
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]  

# Initialize the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the corpus into embeddings
embeddings = model.encode(corpus)

# Save the embeddings using pickle
with open('ng20_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
    
# Save the labels using pickle
with open('ng20_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)