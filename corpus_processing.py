import os

import gensim
import nltk
import numpy as np
import numpy as np
import pandas as pd


def normalize_corpus(papers):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.append("use")
    expand_stopwords_list = ['mm', 'fig', "aa","ef","ciency", "ef ciency", "figure", "also", "ed"]
    for i in expand_stopwords_list:
        stop_words.append(i)
    wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = nltk.stem.wordnet.WordNetLemmatizer()
    norm_papers = []
    for paper in papers:
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers

def read_papers(data_path):

    print(os.listdir(data_path))

    folders = "text"
    papers = []

    file_names = os.listdir(data_path + folders)
    for file_name in file_names:
        with open(data_path + folders + '/' + file_name, encoding='utf-8', errors='ignore', mode='+r') as f:
            data = f.read()
        papers.append(data)
    print(len(papers))
    return papers


if __name__ == "__main__":

    papers = read_papers('nipstxt/')

    norm_papers = normalize_corpus(papers)
    print(len(norm_papers))
    print(norm_papers[0][:50])
    # print(len(norm_papers[0][:50]))

    bigram_model = gensim.models.phrases.Phrases(norm_papers, min_count=20, threshold=20)
    print(bigram_model[norm_papers[0]][:50])
    norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

    dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
    print("sample word to number mappings:", list(dictionary.items())[:15])
    print("total vocabulary size:", len(dictionary))

    dictionary.filter_extremes(no_below=20, no_above=0.6)
    print("total vocabulary size:", len(dictionary))

    bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
    print(bow_corpus[1][:50])

    print([(dictionary[idx], freq) for idx, freq in bow_corpus[1][:50]])

    print("total number of papers:", len(bow_corpus))

    # ----------------LSI------------------------------

    TOTAL_TOPICS = 6
