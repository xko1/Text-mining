import corpus_processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from IPython.display import display
import os
import gensim
import nltk
import numpy as np
import numpy as np
import pandas as pd


cv = CountVectorizer(min_df=20, max_df=0.6, ngram_range=(1, 2), token_pattern=None, tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
papers = corpus_processing.read_papers('nipstxt/')
norm_papers = corpus_processing.normalize_corpus(papers)
# print("prvnich 50", norm_papers[0][:50])


cv_features = cv.fit_transform(norm_papers)
print(cv_features.shape)

vocabulary = np.array((cv.get_feature_names_out()))
print('Total Vocabulary Size:', len(vocabulary))


TOTAL_TOPICS = 20    # TODO Potřeba udělat program pro zjištění idealního počtu témat

lsi_model = TruncatedSVD(n_components=TOTAL_TOPICS, n_iter=500, random_state=42)
document_topics = lsi_model.fit_transform(cv_features)

topic_terms = lsi_model.components_
print(topic_terms.shape)

top_terms = 20
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
topic_key_term_weights = np.array([topic_terms[row, columns] for row, columns in list(zip(np.arange(TOTAL_TOPICS), topic_key_term_idxs))])

topic_keyterms = vocabulary[topic_key_term_idxs]
topic_keyterms_weights = list(zip(topic_keyterms, topic_key_term_weights))
for n in range(TOTAL_TOPICS):
    print("Topic #"+str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    terms, weights = topic_keyterms_weights[n]
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
document_numbers = [13, 250, 500]

for document_number in document_numbers:
    top_topics = list(dt_df.columns[np.argsort(-np.absolute(dt_df.iloc[document_number].values))[:3]])
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:')
    print(papers[document_number][:1000])
    print()

print('='*100)
print('='*46 + 'LDA' + '='*51)

lda_model = LatentDirichletAllocation(n_components=TOTAL_TOPICS, max_iter=500, max_doc_update_iter=50, learning_method='online', batch_size=1740, learning_offset=50., random_state=42)
document_topics = lda_model.fit_transform(cv_features)

topic_terms = lda_model.components_
topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
topic_keyterms = vocabulary[topic_key_term_idxs]
topics = [','.join(topic) for topic in topic_keyterms]
pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame(topics, columns=['Terms per Topic'], index=['Topic'+str(t) for t in range(1, TOTAL_TOPICS+1)])
display(topics_df)