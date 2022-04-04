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

def break_label(text_list, max_length, max_words_per_line = 2, sep = ' '):
    final_list = []
    for text in text_list:
        big_text_flag = False
        updated_word = ''
        if len(text)>=max_length:
            big_text_flag = True
            text_tokens = str.split(text, sep=sep)
            word_count = 0
            for t in text_tokens:
                print(t)
                if word_count != 0:
                    print(word_count)
                    if word_count%max_words_per_line==0:
                        updated_word = updated_word +sep + t + '\n'
                    else:
                        updated_word = updated_word + sep + t
                else:
                    updated_word = t
                    word_count += 1
                    continue
                word_count+=1
        if big_text_flag == True:
            final_list.append(updated_word)
        else:
            final_list.append(text)
    return final_list

def get_similarity_metrix(df, subset =None):
    if subset!=None:
        df = df.loc[df.index.isin(subset)]
    arr_list = []
    for row in df.index:
        row_similarity = [get_cosine_similarity(df.loc[row,:], df.loc[i,:]) for i in df.index]
        arr_list.append(row_similarity)
    df_metrix = pd.DataFrame(np.asarray(arr_list), columns=df.index, index=df.index)
    return  df_metrix

if __name__ == "__main__":
    sns.set(font_scale=1.2)
    embedding_file = np.load('/home/mirza/PycharmProject/Trained Embedding Visualizer/data/codex-m/trained_embedding/entity_embedding-transE-codexM-Uniform.npy')
    entities_dict = pd.read_table('/home/mirza/PycharmProject/Trained Embedding Visualizer/data/codex-m/dictionary_files/entities.dict', header=None)

    data = build_df(embedding_file, entities_dict)

    # list_of_names = ['medical_device', 'drug_delivery_device',
    #                  'research_device', 'research_activity',
    #                  'manufactured_object', 'clinical_drug',
    #                  'molecular_sequence', 'spacial_concept', 'molecular_sequence','language', 'idea_or_concept', 'human', 'mammal', 'food']

    list_of_names = ['German botanist', 'German botanist and author', 'city in Hessen , Germany', 'city in Hesse , Germany',
                     'German Jewish philosopher and theologian', 'German poet , philosopher , historian , and playwright', 'Italian politician and economist', '1933 American Warner Bros musical film']

    fig = plt.figure(figsize=(16,14))
    arr = get_similarity_metrix(data, subset=list_of_names)
    label_text = arr.index
    added_labels = break_label(label_text, max_length=20, max_words_per_line=2, sep=' ')
    arr.index = added_labels
    arr.columns = added_labels

    cmap = sns.light_palette("blue", as_cmap=True)
    sns.heatmap(arr, cmap = cmap)
    #sns.heatmap(arr, robust=True, fmt="f", cmap='RdBu_r', vmin=0, vmax=2)
    plt.xticks(rotation=45)
    plt.tight_layout()

    #plt.title('Trained Embedding Clustering based on FastText')
    fig.savefig("generated_image/codex uniform", dpi=100)
    plt.show()

