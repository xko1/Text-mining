import preprocesing
import pandas as pd
from IPython.display import display
import nltk
from tabulate import tabulate
import numpy as np
import plotly.express as px

x = preprocesing.read_docx('Sample text 2.docx', paragraph=True)
extra_words = ["mm", "fig"]


a = preprocesing.BOW(x, 10)[2]
print(a[0])
for i in a:
        i[0], i[1] = i[1], i[0]

print(a)
print(tabulate(a, headers=['Words', 'Ocurence']))

bag_of_grams = preprocesing.bag_of_n_grams(x, 10, 2, 2)[2]
print(bag_of_grams[0])

for j in bag_of_grams:
        j[1] = j[1] + " " + j[2]
        j[0], j[1] = j[1], j[0]
        j.pop(2)
        print(j)

print(tabulate(bag_of_grams, headers=['Words', 'Ocurence']))
x = x + "three"

print("this is ", x)
tf_idf_matrix = preprocesing.tf_idf_v(x, rm_stop_words=True, not_tokenized=True)[0]
names = preprocesing.tf_idf_v(x, rm_stop_words=True, not_tokenized=True)[2]
print("those are the names", names)
print(tf_idf_matrix)

m = np.sum(tf_idf_matrix, axis=0)
print(m)


# display(pd.DataFrame(m, columns=names))

l = []
for i in range(0, m.size):
        l.append(m[i])


c = []
for i in range(0, m.size):
        c.append(names[i])

n = []
for j in range(0, m.size):
        s = str(l[j]) + " " + str(c[j])
        n.append(s.split())

for element in n:
        element[0] = float(element[0])
n.sort(reverse=True)
print(n[:10])
#

for i in n:
        i[0], i[1] = i[1], i[0]

print(n)
print(tabulate(n[:10], headers=['Words', 'Sum of weights for all occurencies in text']))

maximums_columns = np.max(tf_idf_matrix, axis=0)

l = []
for i in range(0, maximums_columns.size):
        l.append(maximums_columns[i])


c = []
for i in range(0, maximums_columns.size):
        c.append(names[i])

n = []
for j in range(0, maximums_columns.size):
        s = str(l[j]) + " " + str(c[j])
        n.append(s.split())

for element in n:
        element[0] = float(element[0])
n.sort(reverse=True)
print(n[:10])

for i in n:
        i[0], i[1] = i[1], i[0]

print(n)
print(tabulate(n[:10], headers=['Words', 'highest weights']))

tf_idf_v_unpreprocesed_matrix = preprocesing.tf_idf_v_unpreprocesed(x)[0]
names = preprocesing.tf_idf_v_unpreprocesed(x)[2]
m = np.sum(tf_idf_v_unpreprocesed_matrix, axis=0)
print(m)


# display(pd.DataFrame(m, columns=names))

l = []
for i in range(0, m.size):
        l.append(m[i])


c = []
for i in range(0, m.size):
        c.append(names[i])

n = []
for j in range(0, m.size):
        s = str(l[j]) + " " + str(c[j])
        n.append(s.split())

for element in n:
        element[0] = float(element[0])
n.sort(reverse=True)
print(n[:20])
#

for i in n:
        i[0], i[1] = i[1], i[0]

print("matice podobnosti")
sm = preprocesing.tf_idf_cosine_similarity_single_doc(tf_idf_matrix)

fig = px.imshow(sm)
fig.show()
