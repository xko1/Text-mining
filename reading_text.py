import preprocesing

text = preprocesing.read_pdf("sensors-21-04244-v2.pdf")

y = preprocesing.lemmatize(text)
print(len(y))
y = " ".join(y)


text = preprocesing.sentence_tokenization(y)

for i in range(len(text)):
    print(text[i])


y = preprocesing.normalizace_textu(text)
print(" ")
print(type(y))
print("")

y = [string for string in y if 1 != len(string.split())]
y = [string for string in y if string != '']

y = [string for string in y if string != ' ']
for j in y:
    print(j)

