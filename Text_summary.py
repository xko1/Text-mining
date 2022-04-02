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


stop_words = nltk.corpus.stopwords.words('english')

def norm(doc):
    doc = preprocesing.read_docx('sample_text_1.docx', paragraph=True)
    stop_words = nltk.corpus.stopwords.words('english')
    doc = re.sub(r'[^a-za-z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


def low_rank_svd(matrix, count):
    u, s, vt = svds(matrix, count)
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

    Text_for_preprocesing = preprocesing.read_docx('Sample_text_1.docx', paragraph=True)
    Tokenized_text = preprocesing.sentence_tokenization(Text_for_preprocesing)

    Normalized_text = preprocesing.normalizace_textu(Tokenized_text, remove_digits=False)
    # Normalized_text = ' '.join(Normalized_text)
    print(Normalized_text)

    dt_matrix = preprocesing.tf_idf_v(Normalized_text, not_tokenized=False)[0]
    vocab = preprocesing.tf_idf_v(Normalized_text, not_tokenized=False)[2]
    td_matrix = dt_matrix.T
    print(td_matrix.shape)
    display(pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10))

# -------------------------------------------LSA----------------------------------------------------------
    Number_of_sentences = 35
    num_topics = 2
    u, s, vt = low_rank_svd(td_matrix, count=num_topics)
    print(u.shape, s.shape, vt.shape)
    term_topic_mat, singular_values, topic_document_mat = u, s, vt

    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0

    salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
    print(salience_scores)

    top_sentences_indices = (-salience_scores).argsort()[:Number_of_sentences]
    top_sentences_indices.sort()
    print('\n'.join(np.array(Tokenized_text)[top_sentences_indices]))

# --------------------------------------TEXT RANK--------------------------------------------------------------

    similarity_matrix = preprocesing.tf_idf_cosine_similarity_single_doc(dt_matrix)
    similarity_matrix = np.round(similarity_matrix, 3)
    print(similarity_matrix.shape)
    print(similarity_matrix)

    similarity_graph = networkx.from_numpy_array(similarity_matrix, create_using=nx.DiGraph)
    plt.figure(figsize=(12, 6))
    networkx.draw_networkx(similarity_graph)
    print(type(similarity_graph))
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
    print(ranked_sentences[:10])
    plt.show()

    top_sentences_indices = [ranked_sentences[index][1] for index in range(Number_of_sentences)]
    top_sentences_indices.sort()
    print('\n'.join(np.array(Tokenized_text)[top_sentences_indices]))