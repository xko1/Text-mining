import pandas as pd

import preprocesing
import numpy as np
from numpy import dot
from numpy.linalg import norm
#from keras.preprocessing import text, sequence
#from keras.utils import np_utils




if __name__ == "__main__":
    a = "The sky is blue and beautiful. The quick brown fox jumps over the lazy dog. Love this blue and beautiful sky.  A king's breakfast has sausages, ham, bacon, eggs, toast and beans. I love green eggs, ham, sausages and bacon. The brown fox is quick and the blue dog is lazy. The sky is very blue and the sky is very beautiful today. The dog is lazy but the brown fox is quick."
    b = "It was so funny that I couldnt stop laughing really. "
    c = "WTF is going out here"

    norm = preprocesing.sentence_tokenization(a)
    norm = preprocesing.normalizace_textu(norm)
    norm = np.array(norm)
    df_norm = pd.DataFrame({'Document': norm})
    print("norm", norm)
    x = preprocesing.tf_idf_v(a, rm_stop_words=True)

    preprocesing.BOW(a, 10)

    print(type(norm))
    print("matice tf-idf")
    y = preprocesing.tf_idf_v(a)[0]
    yy = preprocesing.tf_idf_v(a)[2]
#   x = preprocesing.tf_idf_v(b)[0]
#   z = preprocesing.tf_idf_v(c)[0]
    print(yy)
    print("matice podobnosti")
    sm = preprocesing.tf_idf_cosine_similarity_single_doc(y)

# -------------------AHC------------------------------
    linkage_matrix = preprocesing.ahc(sm)
    preprocesing.ahc_dendrogram(linkage_matrix)
    preprocesing.use_of_fcluster(linkage_matrix, df_norm)

# -------------------------LDA------------------------
# potřeba matice z BOW
#    preprocesing.lda_model(preprocesing.BOW(a, 10, if_print=False)[0])

    preprocesing.lda_model(a)

