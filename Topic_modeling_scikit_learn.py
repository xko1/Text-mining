import corpus_processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocesing
import os
import gensim
import nltk
import numpy as np
import numpy as np
import pandas as pd
import pyLDAvis.sklearn
import dill
import warnings
from tqdm import tqdm
from wordcloud import WordCloud




def optimal_number_of_topics():
    return 0


cv = CountVectorizer(min_df=0.2, max_df=1.0, ngram_range=(1, 2), token_pattern=None, tokenizer=lambda doc: doc,
                     preprocessor=lambda doc: doc)

tfidf_vectorizer = TfidfVectorizer(min_df=0., max_df=1., norm="l2", use_idf=True, smooth_idf=True)
papers = corpus_processing.read_papers('nipstxt/')
norm_papers = corpus_processing.normalize_corpus(papers)


cv_features = cv.fit_transform(norm_papers)
print(cv_features.shape)

vocabulary = np.array((cv.get_feature_names_out()))


TOTAL_TOPICS = 6

lsi_model = TruncatedSVD(n_components=TOTAL_TOPICS, n_iter=100)
document_topics = lsi_model.fit_transform(cv_features)

topic_terms = lsi_model.components_
print(topic_terms.shape)

top_terms = 20
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
range_list = list(zip(np.arange(TOTAL_TOPICS), topic_key_term_idxs))
print(range_list)
topic_key_term_weights = np.array([topic_terms[row, columns] for row, columns in range_list])
print(topic_key_term_weights)
topic_keyterms = vocabulary[topic_key_term_idxs]
topic_keyterms_with_weights = list(zip(topic_keyterms, topic_key_term_weights))
for n in range(TOTAL_TOPICS):
    print("Topic #"+str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    terms, weights = topic_keyterms_with_weights[n]
    term_weights = sorted([(t, w) for t, w in zip(terms, weights)], key=lambda row: -abs(row[1]))
    for term, wt in term_weights:
        if wt >= 0:
            d1.append((term, round(wt, 3)))
        else:
            d2.append((term, round(wt, 3)))
    print('Direction 1:', d1)
    print('-'*50)
    print('Direction 2:', d2)
    print('-'*50)
    print()

dt_df = pd.DataFrame(np.round(document_topics, 3), columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)])
document_numbers = [0, 1, 2, 3, 4, 5]
# document_numbers = [1]

for document_number in document_numbers:
    top_topics = list(dt_df.columns[np.argsort(-np.absolute(dt_df.iloc[document_number].values))[:3]])
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print()

print('='*100)
print('='*46 + 'LDA' + '='*51)

TOTAL_TOPICS = 6
lda_model = LatentDirichletAllocation(n_components=TOTAL_TOPICS, max_iter=100,
                                      learning_method='online', batch_size=128, n_jobs=4)
document_topics = lda_model.fit_transform(cv_features)
print("Lda preplexity", lda_model.perplexity(cv_features.toarray()))
topic_terms = lda_model.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [','.join(topic) for topic in topic_keyterms]
pd.set_option('display.max_colwidth', None)
topics_df = pd.DataFrame(topics, columns=['Terms per Topic'], index=['Topic'+str(t) for t in range(1, TOTAL_TOPICS+1)])
display(topics_df)

dt_df = pd.DataFrame(document_topics, columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)])
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_colwidth', 200)

max_contrib_topics = dt_df.max(axis=0)
dominant_topics = max_contrib_topics.index

contrib_perc = max_contrib_topics.values
document_number = [dt_df[dt_df[t] == max_contrib_topics.loc[t]].index[0] for t in dominant_topics]

documents = [papers[i] for i in document_numbers]

for document_num in document_numbers:
    top_topics = dt_df.idxmax(axis=1)
    indexes = dt_df.columns
    print(indexes)
    print('Document #'+str(document_num)+':')
    print('Dominant Topics (top 3):',  top_topics[document_num])
    print()

print(dominant_topics)
print(" ")
print(contrib_perc)
print(" ")
print(document_numbers)
print(" ")
print("topics_df\n", topics_df)
print(" ")

results_df = pd.DataFrame({'Dominant Topic': dominant_topics, 'Contribution %': contrib_perc,
                           'Paper Num': document_numbers})
display(results_df)

