import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def build_df(embedding_file, entities_dict):
    embedding_df = pd.DataFrame(embedding_file)
    embedding_df.index = entities_dict[1]
    return embedding_df

def get_cosine_similarity(vec_a, vec_b):
    cos_sim = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
    return cos_sim

def get_diagonal_mask(data):
    temp = np.ones_like(np.array(data), dtype=bool)
    mask = np.ones_like(temp, dtype=bool)
    mask[:,:] = False
    mask[np.diag_indices_from(mask)] = True
    return mask

def break_label(text_list, max_length, max_words_per_line = 1, sep = ' '):
    final_list = []
    for text in text_list:
        big_text_flag = False
        updated_word = ''
        if len(text)>=max_length:
            big_text_flag = True
            text_tokens = str.split(text, sep=sep)
            word_count = 1
            #print(text)
            for t in text_tokens:
                print(word_count)
                if word_count != 1:
                    #print(word_count)
                    if word_count%max_words_per_line==0:
                    #if word_count == 2:
                        #print(t)
                        updated_word = updated_word +sep + t + '\n'
                        word_count += 1
                    else:
                        updated_word = updated_word + sep + t
                        word_count += 1
                else:
                    updated_word = t
                    word_count += 1
                    continue

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
    sns.set(font_scale=1.5)
    embedding_file = np.load('/home/mirza/PycharmProject/Trained Embedding Visualizer/data/codex-m/trained_embedding/entity_embedding-transE-codexM-Uniform.npy')
    #embedding_file

    #scaler = MinMaxScaler()
    #embedding_file = scaler.fit_transform(embedding_file)

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
    #d = np.array(arr)
    # cmap = sns.diverging_palette(100, 7, s=75, l=40,
    #                              n=5, center="light", as_cmap=True)
    #mask = np.triu(np.ones_like(arr, dtype=bool))
    mask = get_diagonal_mask(arr)
    cmap = sns.light_palette("green", as_cmap=True)
    sns.heatmap(arr , mask = mask, fmt='.2f' ,square = True ,cmap = cmap, annot = True)
    #sns.heatmap(arr, robust=True, fmt="f", cmap='RdBu_r', vmin=0, vmax=2)
    plt.xticks(rotation=45)
    plt.tight_layout()

    #plt.title('Trained Embedding Clustering based on FastText')
    fig.savefig("generated_image/codex uni", dpi=100)
    plt.show()

