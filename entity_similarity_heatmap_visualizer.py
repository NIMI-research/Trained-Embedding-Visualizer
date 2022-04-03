import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt

def build_df(embedding_file, entities_dict):
    embedding_df = pd.DataFrame(embedding_file)
    embedding_df.index = entities_dict[1]
    return embedding_df

def get_cosine_similarity(vec_a, vec_b):
    cos_sim = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
    return cos_sim

def get_similarity_metrix(df, subset =None):
    if subset!=None:
        df = df.loc[df.index.isin(subset)]
    arr_list = []
    for row in df.index:
        row_similarity = [get_cosine_similarity(df.loc[row,:], df.loc[i,:]) for i in df.index]
        arr_list.append(row_similarity)
    df_metrix = pd.DataFrame(np.asarray(arr_list), columns=df.index, index=df.index)
    return  df_metrix

embedding_file = np.load('/home/mirza/PycharmProject/Trained Embedding Visualizer/data/codex-m/trained_embedding/codex-m_Embedings-transformer.npy')
entities_dict = pd.read_table('/home/mirza/PycharmProject/Trained Embedding Visualizer/data/codex-m/dictionary_files/entities.dict', header=None)

data = build_df(embedding_file, entities_dict)
# list_of_names = ['medical_device', 'drug_delivery_device',
#                  'research_device', 'research_activity',
#                  'manufactured_object', 'clinical_drug',
#                  'molecular_sequence', 'spacial_concept', 'molecular_sequence','language', 'idea_or_concept', 'human', 'mammal', 'food']

list_of_names = ['German botanist', 'German botanist and author', 'city in Hessen , Germany', 'city in Hesse , Germany',
                 'German Jewish philosopher and theologian', 'German poet , philosopher , historian , and playwright', 'Italian politician and economist']

fig = plt.figure(figsize=(14,12))
arr = get_similarity_metrix(data, subset=list_of_names)
cmap = sns.light_palette("blue", as_cmap=True)
sns.heatmap(arr, cmap = cmap)
#sns.heatmap(arr, robust=True, fmt="f", cmap='RdBu_r', vmin=0, vmax=2)
plt.xticks(rotation=90)
plt.tight_layout()

#plt.title('Trained Embedding Clustering based on FastText')
fig.savefig("generated_image/codex_transformer", dpi=100)
plt.show()

