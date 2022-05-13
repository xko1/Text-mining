import networkx as nx
import nltk
import numpy as np
import re
import preprocesing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from IPython.display import display
from scipy.sparse.linalg import svds
import networkx
import matplotlib.pyplot as plt
from tqdm import tqdm




def low_rank_svd(matrix, count):
    u, s, vt = svds(matrix, k=count)
    return u, s, vt


def my_norm(doc):
    Norm = preprocesing.normalizace_textu(doc)
    return ' '.join(Norm)





if __name__ == "__main__":


    # stop_words = nltk.corpus.stopwords.words('english')
    # doc = re.sub(r'[^a-za-z\s]', '', doc, re.I|re.A)
    # doc = doc.lower()
    # doc = doc.strip()
    # tokens = nltk.word_tokenize(doc)
    # filtered_tokens = [token for token in tokens if token not in stop_words]
    # doc = ' '.join(filtered_tokens)
    # print(type(doc))

    Text_for_preprocesing = preprocesing.read_docx('sample text 2.docx', paragraph=True)
    tokenized_text = preprocesing.sentence_tokenization(Text_for_preprocesing)

    norm = preprocesing.normalizace_textu(tokenized_text, remove_digits=True, stopword_removal=False)
    print('this is norm', norm)
    # norm = [string for string in norm if len(string.split()) != 1]
    # norm = [string for string in norm if string != '']
    norm = [string for string in norm if string != ' ']
    norm = [string for string in norm if string != '  ']
    for i in norm:
        print(i)
    print(norm)
    dt_matrix = preprocesing.tf_idf_v_lsa(Text_for_preprocesing, rm_stop_words=True, not_tokenized=True)[0]
    vocab = preprocesing.tf_idf_v_lsa(Text_for_preprocesing, rm_stop_words=True, not_tokenized=True)[2]
    preprocessed_text = preprocesing.tf_idf_v_lsa(Text_for_preprocesing, rm_stop_words=True, not_tokenized=True)[3]
    td_matrix = dt_matrix.T
    print(td_matrix.shape)
    display(pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10))

# -------------------------------------------LSA----------------------------------------------------------
    Number_of_sentences = 12
    num_topics = 30
    u, s, vt = low_rank_svd(td_matrix, count=num_topics)
    print(u.shape, s.shape, vt.shape)
    term_topic, singular_val, topic_document = u, s, vt
    print("singular values", max(singular_val))
    print("singular values", singular_val)

    sv_threshold = 0.5
    min_sigma_val = max(singular_val) * sv_threshold
    singular_val[singular_val < min_sigma_val] = 0

    salience_scores = np.sqrt(np.dot(np.square(singular_val), np.square(topic_document)))
    print(salience_scores)

    top_sentences_indices = (-salience_scores).argsort()[:Number_of_sentences]
    print('top sentences indices \n')
    top_sentences_indices.sort()
    print(top_sentences_indices)

    print('this is LSA', '\n'.join(np.array(preprocessed_text)[top_sentences_indices]))

# --------------------------------------TEXT RANK--------------------------------------------------------------

    similarity_matrix = preprocesing.tf_idf_cosine_similarity_single_doc(dt_matrix)
    similarity_matrix = np.round(similarity_matrix, 3)
    print(similarity_matrix.shape)
    print(similarity_matrix)

    similarity_graph = networkx.from_numpy_array(similarity_matrix, create_using=nx.Graph)
    similarity_graph.remove_edges_from(nx.selfloop_edges(similarity_graph))
    plt.figure(figsize=(12, 6))
    networkx.draw_networkx(similarity_graph)
    print(type(similarity_graph))
    scores = networkx.pagerank(similarity_graph)
    print(scores)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    print(ranked_sentences[:10])

    top_sentences_indices = [ranked_sentences[index][1] for index in range(Number_of_sentences)]
    top_sentences_indices.sort()
    print(top_sentences_indices)

    print('\n'.join(np.array(preprocessed_text)[top_sentences_indices]))
    plt.show()

