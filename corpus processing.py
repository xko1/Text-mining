import os
import numpy as np
import pandas as pd
import preprocesing
import nltk
import gensim
import time


def normalize_corpus(papers):
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

if __name__ == "__main__":

    DATA_PATH = 'nipstxt/'
    print(os.listdir(DATA_PATH))

    folders = ["nips{0:02}".format(i) for i in range(0, 13)]
    papers = []
    for folder in folders:
        file_names = os.listdir(DATA_PATH + folder)
        for file_name in file_names:
            with open(DATA_PATH + folder + '/' + file_name, encoding='utf-8', errors='ignore', mode='+r')as f:
                data = f.read()
            papers.append(data)
    print(len(papers))

    # print((papers[0][:10000]))

    stop_words = nltk.corpus.stopwords.words('english')
    wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = nltk.stem.wordnet.WordNetLemmatizer()

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

    TOTAL_TOPICS = 10
    lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary, num_topics=TOTAL_TOPICS, onepass=True, chunksize=1740, power_iters=1000)

    for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
        print("topic #" +str(topic_id+1)+":")
        print(topic)
        print()

    for n in range(TOTAL_TOPICS):
        print("topic #" +str(n+1)+":")
        print("="*50)
        d1 = []
        d2 = []
        for term, wt in lsi_bow.show_topic(n, topn=20):
            if wt >= 0:
                d1.append((term, round(wt, 3)))
            else:
                d2.append((term, round(wt, 3)))
        print("Direction 1:", d1)
        print('-'*50)
        print("Direction 2:", d2)
        print("-"*50)
        print()

    term_topic = lsi_bow.projection.u
    singular_values = lsi_bow.projection.s
    topic_document = (gensim.matutils.corpus2dense(lsi_bow[bow_corpus], len(singular_values)).T / singular_values).T
    print(term_topic.shape, singular_values.shape, topic_document.shape)


    document_topics = pd.DataFrame(np.round(topic_document.T, 3), columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)])
    document_topics.head(5)
    # preprocesing.display(document_topics)

    document_numbers = [13, 250, 500]
    for documet_number in document_numbers:
        top_topics = list(document_topics.columns[np.argsort(-np.absolute(document_topics.iloc[documet_number].values))[:3]])
        print("Document #"+str(document_numbers)+':')
        print("Dominant Topics (top 3): ", top_topics)
        print("Paper Summary:")
        print(papers[documet_number][:500])
        print()

    # print(papers[13])

# -----------------------------------LDA-------------------------------------------

    lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1740, alpha='auto', eta='auto', random_state=42,iterations=500, num_topics=TOTAL_TOPICS, passes=20, eval_every=None)

    for topic_id, topic in lda_model.print_topics(num_topics=10, num_words=20):
        print('Topic #'+str(topic_id+1)+':')
        print(topic)
        print()

    topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
    avg_coherence_score = np.mean([item[1] for item in topics_coherences])
    print('Avg. Coherence Score:', avg_coherence_score)