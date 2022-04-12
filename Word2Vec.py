import nltk
import preprocesing
from gensim.models import word2vec


if __name__ == "__main__":
    doc = preprocesing.read_docx('Sample_text_1.docx', paragraph=True)
    print(doc)
    # tokenized_text = preprocesing.word_tokenization(doc)
    tokenized_text = preprocesing.sentence_tokenization(doc)
    print(tokenized_text)
    preprocesd_text = preprocesing.normalizace_textu(tokenized_text)

    with open('preprocesd_text.txt', 'w') as text_file:
        text_file.writelines('\n'.join(preprocesd_text))
    text_file.close()


    word_to_vec = word2vec.Word2Vec(vector_size=100, window=50, min_count=5, sample=1e-3, epochs=50)
    word_to_vec.build_vocab(corpus_file='preprocesd_text.txt')
    word_to_vec.train(corpus_file='preprocesd_text.txt', epochs=word_to_vec.epochs, total_examples=word_to_vec.corpus_count, total_words=word_to_vec.corpus_total_words)
    input_words = ['field', 'structure', 'magnetic', 'coil', 'result']
    print(word_to_vec)

    print(word_to_vec.wv.most_similar('field', topn=5))

    # similar_words = {search_term: [item[0] for item in word_to_vec.wv.most_similar(search_term, topn=5)] for search_term in input_words}
    # print(similar_words)

    # similar_word = word_to_vec.wv.most_similar('u', topn=5)
    # print(word_to_vec.wv.index_to_key)
    #print(similar_word)


