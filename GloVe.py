import numpy as np
import pandas as pd
import sklearn.cluster
import spacy
import preprocesing
from IPython.display import display
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == "__main__":
    text = preprocesing.read_docx('Sample_text_1.docx', paragraph=True)
    print(text)

    #y = preprocesing.normalizace_textu(text)
    print(" ")

    print("")
    #y = [string for string in y if string != '']


    y = preprocesing.lemmatize(text)
    print(len(y))
    print("after lemma", y)
    y = " ".join(y)
    print(y)

    y = preprocesing.sentence_tokenization(y)
    y = preprocesing.normalizace_textu(y)
    y = [string for string in y if string != '']
    for j in range(len(y)):
        print(y[j])
    norm_text = y
    print(norm_text)

    df_norm = pd.DataFrame({'Document': norm_text})
    nlp = spacy.load("en_core_web_lg")
    total_vectors = len(nlp.vocab.vectors)
    print('Total word vectors:', total_vectors)

    unique_words = list(set([word for sublist in [doc.split() for doc in norm_text] for word in sublist]))
    word_glove_vectors = np.array([nlp(word).vector for word in unique_words])
    display((pd.DataFrame(word_glove_vectors, index=unique_words)))

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(word_glove_vectors)
    labels = unique_words

    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

    doc_glove_vector = np.array([nlp(str(doc)).vector for doc in norm_text])

    km = KMeans(n_clusters=10, random_state=0)
    km.fit_transform(doc_glove_vector)
    cluster_labels = km.labels_
    cluster_labels = pd.DataFrame(cluster_labels, columns=['Clusterlabel'])
    display(pd.concat([df_norm, cluster_labels], axis=1))


    plt.show()