import nltk
import preprocesing
from gensim.models import word2vec


if __name__ == "__main__":
    doc = preprocesing.read_docx('Sample_text_1.docx', paragraph=True)
    print(doc)
    # tokenized_text = preprocesing.word_tokenization(doc)
    tokenized_text = preprocesing.sentence_tokenization(doc)
    print(tokenized_text)
    feature_size = 100  # Dimenze vektoru
    window_context = 30  # Velikost okna
    min_word_count = 1  # Minimalní počet slov
    sample = 1e-3
    input_words = ['e']

    word_to_vec = word2vec.Word2Vec(tokenized_text, vector_size=feature_size, window=window_context, min_count=min_word_count, sample=sample, epochs=50)
    print(word_to_vec.wv.index_to_key)
    similar_words = {search_term: [item[0] for item in word_to_vec.wv.most_similar(search_term, topn=5)] for search_term in input_words}
    print(similar_words)
    word_to_vec.wv.key_to_index()
    # similar_word = word_to_vec.wv.most_similar('u', topn=5)
    # print(word_to_vec.wv.index_to_key)
    #print(similar_word)


