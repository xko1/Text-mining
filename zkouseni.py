import pandas as pd

import preprocesing
import numpy as np
import plotly.express as px

from numpy import dot
from numpy.linalg import norm
#from keras.preprocessing import text, sequence
#from keras.utils import np_utils

extra_words = []


if __name__ == "__main__":
    a = "The sky is blue and beautiful. The quick brown fox jumps over the lazy dog. Love this blue and beautiful sky.  A king's breakfast has sausages, ham, bacon, eggs, toast and beans. I love green eggs, ham, sausages and bacon. The brown fox is quick and the blue dog is lazy. The sky is very blue and the sky is very beautiful today. The dog is lazy but the brown fox is quick."
    b = "It was so funny that I couldnt stop laughing really. "
    c = "WTF is going out here"

    a = preprocesing.read_docx('Sample text 2.docx', paragraph=True)
    norm = preprocesing.sentence_tokenization(a)
    norm = preprocesing.normalizace_textu(norm)
    norm = np.array(norm)

    # print("norm", norm)
    x = preprocesing.tf_idf_v(a, rm_stop_words=True, not_tokenized=True)
    # extra_words = ["mm", "fig"]
    # preprocesing.BOW(a, 10)

    print(type(norm))
    print("matice tf-idf")
    y = preprocesing.tf_idf_v(a, rm_stop_words=True, not_tokenized=True)[0]
    yy = preprocesing.tf_idf_v(a,rm_stop_words=True, not_tokenized=True)[2]
    yyy = yy = preprocesing.tf_idf_v(a,rm_stop_words=True, not_tokenized=True)[3]
    df_norm = pd.DataFrame({'Document': yyy})
#   x = preprocesing.tf_idf_v(b)[0]
#   z = preprocesing.tf_idf_v(c)[0]
    print(yy)
    print("matice podobnosti")
    sm = preprocesing.tf_idf_cosine_similarity_single_doc(y)

# -------------------AHC------------------------------
    distance = preprocesing.cosine_distance(sm)
    linkage_matrix = preprocesing.ahc(distance)
    preprocesing.use_of_fcluster(linkage_matrix, df_norm)
    preprocesing.ahc_dendrogram(linkage_matrix, save_figure=False)

# -------------------------LDA------------------------
# potřeba matice z BOW
#    preprocesing.lda_model(preprocesing.BOW(a, 10, if_print=False)[0])
    print("this is LDA")
    preprocesing.lda_model(a)

    Text_for_similariy = """text 2 we discuss the modeling of the magnetic fields that surround nonferromagnetic materials.
in relevant applications the reaction field formed by inserting the examined sample in a homogeneous field facilitates the calculation of the magnetic susceptibility.
magnetic susceptibility is a physical quantity describing material properties in an external magnetic field.
the quantity is defined as the ratio between the magnetization m of a material in a magnetic field and the field intensity h all materials can be classified into three groups by the magnetic susceptibility value.
inserting a specimen with magnetic susceptibility s causes local deformation of a previously homogeneous magnetic field fig.
the behaviour of the magnetic flux density bzx in position y  0 and z  0 on a straight line is shown in figure 1.
  the magnetic flux density inside the specimen will be equal to assume a constant magnetic flux through the normal area of the crosssection sz of the working space of the given magnet.
we have from which it is evident that the magnetic flux density outside the specimen is changed resulting in a shape that can be considered the superposition of the homogeneous field b0 and the reaction field b.
the value b  bz_emnc 47 is plotted on the y axis.
the  is utilized for calculating the susceptibility of paramagnetic substances while the enables us to obtain the susceptibility of diamagnetic ones.
in the table below we summarize the results of the magnetic susceptibility of all the samples.
the paper characterizes a simplified procedure and a formula to calculate magnetic susceptibility 10 both based on eq.
sample text 2 we discuss the modeling of the magnetic fields that surround nonferromagnetic materials.
samples of paramagnetic and diamagnetic materials such as aluminum and copper are are placed in a homogeneous field within an air filled block the samples are spherical cylindrical and cuboidal.
the quantity is defined as the ratio between the magnetization m of a material in a magnetic field and the field intensity h all materials can be classified into three groups by the magnetic susceptibility value.
inserting a specimen with magnetic susceptibility s causes local deformation of a previously homogeneous magnetic field fig.
the behaviour of the magnetic flux density bzx in position y  0 and z  0 on a straight line is shown in figure 1.
  the magnetic flux density inside the specimen will be equal to assume a constant magnetic flux through the normal area of the crosssection sz of the working space of the given magnet.
we have from which it is evident that the magnetic flux density outside the specimen is changed resulting in a shape that can be considered the superposition of the homogeneous field b0 and the reaction field b.
experiment the space configuration of the sample is represented in fig.
the value b  bz_emnc 47 is plotted on the y axis.
the  is utilized for calculating the susceptibility of paramagnetic substances while the enables us to obtain the susceptibility of diamagnetic ones.
in the table below we summarize the results of the magnetic susceptibility of all the samples.
the paper characterizes a simplified procedure and a formula to calculate magnetic susceptibility 10 both based on eq.
"""
    print("Text similarity of text sumarization")
    preprocesd_text = preprocesing.tf_idf_v(Text_for_similariy, rm_stop_words=True, not_tokenized=True)[0]

    similarity_matrix = preprocesing.tf_idf_cosine_similarity_single_doc(preprocesd_text)
    # fig = px.imshow(similarity_matrix)


text = """the paper describe the procedure to verify the validity of the general relationship for calculate the susceptibility from the reaction field for sample of various shape and material . 
the quantity be define as the ratio between the magnetization m of a material in a magnetic field and the field intensity h  all material can be classify into three group by the magnetic susceptibility value . 
the magnitude of such a deformation depend on the difference between the magnetic susceptibility of the sample  s  and its vicinity  v   the volume and shape of the sample  and the magnitude of the basic field b . 
the difference between a change in the magnetic field in the vicinity of the specimen and the value of the static magnetic field b be refer to as the reaction field b .
we have from which it be evident that the magnetic flux density outside the specimen be change  result in a shape that can be consider the superposition of the homogeneous field b and the reaction field b .
   where bv be the reaction field in the vicinity of the specimen  vs be the volume of the specimen  and b be the static magnetic field . 
for an irrotational field  the above equation    assume the form the material relation be then outline by equation the model reaction field b of the sample be show in the cross  section plane  fig . 
to calculate the magnetic susceptibility  we employ the course of the reaction field in the x axis . 
the magnetic susceptibility of the model material be then equal to where the sign   in front of the fraction depend on the material use . 
the value bmax and bmin represent the value of b in the space of the sample  with bmax denote the field close to the inner side of the boundary of the sample and bmin express the field close to the corresponding outer side . 
the magnetic susceptibility of the simulate sample be calculate by use the value of the reaction field b from fig . 
base on model the magnetic field in the vicinity of sample of non  ferromagnetic material  we consider the formula    applicable to all the paramagnetic and diamagnetic material sample .
 this is LSA the paper describe the procedure to verify the validity of the general relationship for calculate the susceptibility from the reaction field for sample of various shape and material . 
the quantity be define as the ratio between the magnetization m of a material in a magnetic field and the field intensity h  all material can be classify into three group by the magnetic susceptibility value . 
the magnitude of such a deformation depend on the difference between the magnetic susceptibility of the sample  s  and its vicinity  v   the volume and shape of the sample  and the magnitude of the basic field b . 
the difference between a change in the magnetic field in the vicinity of the specimen and the value of the static magnetic field b be refer to as the reaction field b .
   where bv be the reaction field in the vicinity of the specimen  vs be the volume of the specimen  and b be the static magnetic field .
for an irrotational field  the above equation    assume the form the material relation be then outline by equation the model reaction field b of the sample be show in the cross  section plane  fig . 
to calculate the magnetic susceptibility  we employ the course of the reaction field in the x axis . 
the magnetic susceptibility of the model material be then equal to where the sign   in front of the fraction depend on the material use . 
the value bmax and bmin represent the value of b in the space of the sample  with bmax denote the field close to the inner side of the boundary of the sample and bmin express the field close to the corresponding outer side . 
the magnetic susceptibility of the simulate sample be calculate by use the value of the reaction field b from fig . 
in the table below we summarize the result of the magnetic susceptibility of all the sample .
base on model the magnetic field in the vicinity of sample of non  ferromagnetic material  we consider the formula    applicable to all the paramagnetic and diamagnetic material sample ."""
preprocesd_text = preprocesing.tf_idf_v(text, rm_stop_words=True, not_tokenized=True)[0]

similarity_matrix2 = preprocesing.tf_idf_cosine_similarity_single_doc(preprocesd_text)
fig2 = px.imshow(similarity_matrix2)
fig2.show()