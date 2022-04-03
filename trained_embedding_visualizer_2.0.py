import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
import seaborn as sns
from scipy.spatial import ConvexHull
from adjustText import adjust_text


def draw_convex_haul(dfs, palette):
    '''
    based on the convex haul created by: https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489
    :param dfs: dataframe to be provided
    :param palette: color palatte
    :return:
    '''
    i = 0
    for df in dfs:
        points = df[['x-axis', 'y-axis']].values
        convex_hull = ConvexHull(points)
        x_hull = np.append(points[convex_hull.vertices, 0],
                           points[convex_hull.vertices, 0][0])
        y_hull = np.append(points[convex_hull.vertices, 1],
                           points[convex_hull.vertices, 1][0])
        plt.fill(x_hull, y_hull, alpha=0.3, c=palette[i])
        i+=1

def fetch_random_indexes_per_cluster(dfs, number_of_samples_per_df = 100):
    random_indexes_overall = []
    for df in dfs:
        random_indexes_per_df = np.random.choice(list(df.index), size=number_of_samples_per_df, replace=False)
        #print(random_indexes_per_df)
        random_indexes_overall.extend(random_indexes_per_df)
    #exit()
    return  random_indexes_overall

def annotate_members(df, random_indexes_to_annotate, palette, x_axis = 'x-axis', y_axis = 'y-axis', text_column = 'text', annotate_specific_labels = False, additional_members = None, additional_member_color = 'black', already_annotated_text=['dummy']):
    overal_texts = []
    overal_texts_2 = []
    annotated_texts = []
    random_indexes_to_annotate = list(random_indexes_to_annotate)
    additional_member_df_indexes = None
    if annotate_specific_labels == True:
        additional_member_df_indexes = list(df.loc[df[text_column].isin(additional_members)].index)
        random_indexes_to_annotate.extend(additional_member_df_indexes)
        random_indexes_to_annotate = np.unique(random_indexes_to_annotate)

    for i in random_indexes_to_annotate:
        if (additional_member_df_indexes!=None) and (annotate_specific_labels!=False) and (additional_members!=None):
            if i in additional_member_df_indexes:
                if  (df[text_column][i] not in already_annotated_text):
                    annotated_texts.append(df[text_column][i])
                    #print('here')
                    overal_texts.append(plt.text(df[x_axis][i], df[y_axis][i], df[text_column][i], color=additional_member_color, fontsize=30,wrap=True))
                else:
                    continue
                continue
        #16,13
        if  (df[text_column][i] not in already_annotated_text):
            annotated_texts.append(df[text_column][i])
            overal_texts.append(plt.text(df[x_axis][i], df[y_axis][i], df[text_column][i], fontsize= 14))
        else:
            continue
    #print(overal_texts)
    #ultimate_text = np.unique([*overal_texts, *overal_texts_2])
    #adjust_text(overal_texts, expand_text = (0.20,0.20), expand_objects = (0.20,0.20))
    #adjust_text(overal_texts_2, expand_text=(0.5,0.5))
    #adjust_text(overal_texts, arrowprops=dict(arrowstyle='->', color='red'), expand_text=(2, 1.5))
    adjust_text(overal_texts, arrowprops=dict(arrowstyle='->', color='red'), expand_text=(0.3, 2.5))
    return annotated_texts
    #return overal_texts


def annotate_centroids(centroids, palette):
    i = 1
    j = 0
    for centroid in centroids:
        text = 'centroid ' + str(i)
        plt.text(centroid[0], centroid[1], 'x' , horizontalalignment='left', size=20,
             color=palette[j], weight='bold')
        i+=1
        j+=1

def reduce_dimension(alg_type = 'pca', dim = 2):
    init_var = None,
    if alg_type == 'pca':
        init_var = PCA(dim)
    elif alg_type == 'TSNE':
        init_var = TSNE(n_components=dim, learning_rate='auto', init = 'random')
    elif alg_type == 'ISOMAP':
        init_var = Isomap(n_components=dim)
    elif alg_type == 'SpectralEmbedding':
        init_var = SpectralEmbedding(n_components=dim)
    else:
        print('dimensionality reduction technique is not listed')
        exit()
    return init_var


def scatter_plot(x, y, data, hue, n_clusters, palette ,title, xlabel, ylabel):
    palette_subset = [palette[i] for i in range(n_clusters)]
    p1 = sns.scatterplot(x, y, data=data, hue=hue, palette=palette_subset, s = 80)
    plt.title(title)

    # for i in range(n_clusters):
    #     current_label = 'cluster ' + str(i)
    #     for t, l in zip(p1._legend.texts, current_label):
    #         t.set_text(l)
    #p1._legend(['A','B', 'C', 'D', 'E'])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

if __name__ == "__main__":
    #TODO input the embeddings, entity to id file
    trained_emb_dir = '/home/mirza/PycharmProject/Trained Embedding Visualizer/data/umls/trained_embeddings/umls_entity_embedding-SANS-KMEANS-Hita1.npy'
    #trained_emb_2_dir = '/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/umls/trained_embedding/sans-kmeans-pretrained (1).npy'
    entity_to_id_dict_dir = '/home/mirza/PycharmProject/Trained Embedding Visualizer/data/umls/dictionary_files/entities.dict'
    #TODO input the embedding type: 1. pca, 2. TSNE, 3. ISOMAP 4. SpectralEmbedding
    alg_type = 'TSNE'
    #TODO input the number of text samples in the graph per cluster
    n_text = 15
    #TODO input the number of clusters
    n_clusters = 6
    #TODO input the color pallete
    palette = sns.color_palette("tab10")

    #TODO if we want to use same random numbers!
    np.random.seed(42)

    sns.set(font_scale=2)

    #TODO image title
    image_title = 'Trained Embedding Visualization for From Our Approach'

    # trained_Emb = torch.load('/home/mirza/PycharmProjects/pythonProject1/Negative-Sampling_LM/SANS-master-KMeans/data/mahfuza apu data/MTE_FS_entity.pkl', map_location=torch.device('cpu'))
    # trained_Emb = pd.DataFrame(trained_Emb.weight.detach().numpy())

    trained_Emb=np.load(trained_emb_dir)
    entity_to_id = pd.read_table(entity_to_id_dict_dir, header=None)
    entity_to_id_text = entity_to_id[1].values


    #Reduce reduce the embeddings into lower dimension, currently only 2d visualization is possible
    trained_Emb_low_d = reduce_dimension(alg_type=alg_type, dim=2).fit_transform(trained_Emb)

    #initialize kmeans
    kmeans = KMeans(n_clusters=n_clusters,init='k-means++', n_init=10, max_iter=1000, verbose=0, random_state=1234)

    # predict the labels of clusters.
    label = kmeans.fit_predict(trained_Emb_low_d)
    centroids = kmeans.cluster_centers_

    unique_label = np.unique(label)
    info_array = np.c_[trained_Emb_low_d,label,entity_to_id_text]

    info_df = pd.DataFrame(info_array)



    info_df.columns = ['x-axis', 'y-axis', 'label', 'text']

    per_class_entities = []
    class_number = []
    for i in unique_label:
        class_i_members = info_df.loc[info_df['label']==i]['text'].unique()
        class_i_member_size = len(info_df.loc[info_df['label']==i])
        per_class_entities.append(class_i_members)
        class_number.append(i)
        #print(i, class_i_member_size, class_i_members)

    class_member_df = np.c_[class_number, per_class_entities]
    print(class_member_df)
    #pd.DataFrame(class_member_df).to_csv()


    individual_df_per_cluster = [info_df.loc[info_df['label']==i] for i in unique_label]
    #random_indexes_to_annotate = fetch_random_indexes_per_cluster(individual_df_per_cluster, number_of_samples_per_df=n_text)
    random_indexes_to_annotate = np.random.choice(list(info_df.index), size=n_text, replace=False)
    # additional_members = ['biomedical_occupation_or_discipline', 'body_location_or_region', 'event'
    #                     'functional_concept', 'geographic_area', 'laboratory_or_test_result',
    #                     'occupation_or_discipline', 'organism_attribute', 'physical_object',
    #                     'qualitative_concept', 'quantitative_concept', 'sign_or_symptom',
    #                     'substance']

    # additional_members = ['activity', 'age_group', 'anatomical_structure', 'behavior'
    #  'body_location_or_region' ,'body_part_organ_or_organ_component',
    #  'body_substance', 'cell', 'cell_component', 'cell_function', 'classification',
    #  'clinical_attribute' ,'daily_or_recreational_activity',
    #  'diagnostic_procedure' ,'educational_activity', 'embryonic_structure',
    #  'entity', 'environmental_effect_of_humans', 'event', 'finding',
    #  'fully_formed_anatomical_structure', 'gene_or_genome', 'geographic_area',
    #  'governmental_or_regulatory_activity', 'group_attribute',
    #  'health_care_activity', 'health_care_related_organization',
    #  'human_caused_phenomenon_or_process', 'individual_behavior',
    #  'intellectual_product', 'laboratory_or_test_result', 'laboratory_procedure',
    #  'language', 'machine_activity', 'molecular_biology_research_technique',
    #  'molecular_sequence', 'occupational_activity', 'organism_attribute',
    #  'organization' ,'patient_or_disabled_group', 'phenomenon_or_process',
    #  'physical_object', 'population_group', 'professional_or_occupational_group',
    #  'professional_society', 'qualitative_concept', 'quantitative_concept',
    #  'regulation_or_law', 'research_activity',
    #  'self_help_or_relief_organization', 'sign_or_symptom', 'social_behavior',
    #  'spatial_concept', 'substance', 'therapeutic_or_preventive_procedure',
    #  'tissue']

    #additional_members= ['group_attribute', 'classification', 'clinical_device', 'self_help_or_relief_organization', 'regulation_or_law', 'governmental_or_regulatory_activity']
    additional_members = ['clinical_drug', 'drug_delivery_device', 'medical_device',
           'manufactured_object', 'research_device']
    additional_members_2 = [
         'idea_or_concept', 'spatial_concept',
         'molecular_sequence',
          'language']

    #intellectual_product, functional_concept

    # additional_members = ['biomedical_occupation_or_discipline',
    #                       'occupation_or_discipline',
    #                      'sign_or_symptom',
    #                       'substance']
    #print(random_indexes_to_annotate)
    #TODO label label names, keep one
    #TODO overlapping one should be removed
    #TODO axis labels needs to be removed
    #TODO keep only one legend
    fig = plt.figure(figsize=(14,12))
    x_axis = alg_type + '_1'
    y_axis = alg_type + '_2'
    p1 = scatter_plot('x-axis', 'y-axis',
                 data = info_df,
                 hue = 'label',
                 n_clusters=n_clusters,
                 palette = palette,
                 title = image_title,
                 xlabel = '',
                 ylabel = '')

    current_labels = ['cluster ' + str(i) for i in range(n_clusters)]
    for i in range(n_clusters):
        #current_label = 'cluster ' + str(i)
        for t, l in zip(p1.legend_.texts, current_labels):
            t.set_text(l)
    new_title = 'Obtained Clusters'
    p1.legend_.set_title(new_title)

    #plt.legend(labels=["A", "B", "C", "D", "E"], title="clusters")

    already_annotated_text = annotate_members(info_df, random_indexes_to_annotate, palette, 'x-axis', 'y-axis', 'text', annotate_specific_labels=True, additional_members= additional_members, additional_member_color='green')
    additional_members_2 = list([*already_annotated_text, *additional_members_2])
    #print(additional_members_2)
    already_annotated_text_second = annotate_members(info_df, random_indexes_to_annotate, palette, 'x-axis', 'y-axis', 'text',
                                     annotate_specific_labels=True, additional_members=additional_members_2, additional_member_color='red', already_annotated_text=already_annotated_text)
    #annotate_centroids(centroids, palette)
    draw_convex_haul(dfs=individual_df_per_cluster, palette=palette)
    #fig = plt.gcf()
    #plt.legend()
    p1.legend_.remove()
    #plt.legend(labels=["A", "B", "C", "D", "E", "F"], title="clusters")
    plt.tight_layout()
    plt.show()
    p1.set(xlabel=None)
    p1.set(ylabel=None)

    plt.title('Trained Embedding Clustering based on FastText')
    #fig.set_size_inches(8, 6)
    fig.savefig("generated_image/Trained_embedding_KNS", dpi=100)

