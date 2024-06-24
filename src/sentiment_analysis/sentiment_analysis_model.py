import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.metrics import ConfusionMatrix

TRAIN_CSV_PATH = "/csv/train.csv"
TEST_CSV_PATH = "/csv/test.csv"
YT_COMMENTS_PATH = "/csv/youtube_comments.csv"

exemplo_base = pd.read_csv(TRAIN_CSV_PATH)
exemplo_base.columns = ['Phrase','Emotion']
print(exemplo_base.head())

exemplo_base_teste = pd.read_csv(TEST_CSV_PATH)
exemplo_base_teste.columns = ['Phrase','Emotion']

print('Tamanho da base de Treino {}'.format(exemplo_base.shape[0]))
print(exemplo_base.Emotion.value_counts())

print((exemplo_base.Emotion.value_counts()/exemplo_base.shape[0])*100)

import pandas as pd

seed = 42

joy_sample = exemplo_base[exemplo_base['Emotion'] == 'joy'].sample(n=500, random_state=seed)
sadness_sample = exemplo_base[exemplo_base['Emotion'] == 'sadness'].sample(n=500, random_state=seed)
anger_sample = exemplo_base[exemplo_base['Emotion'] == 'anger'].sample(n=500, random_state=seed)
fear_sample = exemplo_base[exemplo_base['Emotion'] == 'fear'].sample(n=500, random_state=seed)
love_sample = exemplo_base[exemplo_base['Emotion'] == 'love'].sample(n=500, random_state=seed)
surprise_sample = exemplo_base[exemplo_base['Emotion'] == 'surprise'].sample(n=500, random_state=seed)

new_dataset = pd.concat([joy_sample, sadness_sample, anger_sample, fear_sample, love_sample, surprise_sample])

new_dataset = new_dataset.reset_index(drop=True)

print('Tamanho da base balanceada {}'.format(new_dataset.shape[0]))

print(new_dataset.Emotion.value_counts())

new_dataset.sample(n=20)

lista_Stop = nltk.corpus.stopwords.words('english')
np.transpose(lista_Stop)

def removeStopWords(texto):
    frases = []
    for (palavras, sentimento) in texto:
        # Create a list comprehension to extract only the words that are not in lista_Stop
        semStop = [p for p in palavras.split() if p.lower() not in lista_Stop]
        # Insert the phrases with the labels (sentimento) already treated by lista_Stop
        frases.append((semStop, sentimento))
    return frases

def aplica_Stemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    # RSLPStemmer is specific to the Portuguese language
    frases_semStemming = []
    for (palavras, sentimento) in texto:
        com_stemming = [str(stemmer.stem(p)) for p in palavras.split() if p.lower() not in lista_Stop]
        frases_semStemming.append((com_stemming, sentimento))
    return frases_semStemming

new_list = list(zip(new_dataset['Phrase'], new_dataset['Emotion']))
new_list_teste = list(zip(exemplo_base_teste['Phrase'], exemplo_base_teste['Emotion']))
#print(new_list)

frases_com_Stem_treino = aplica_Stemmer(new_list)

pd.DataFrame(frases_com_Stem_treino, columns=['Phrase','Emotion']).sample(10)

frases_com_Stem_teste = aplica_Stemmer(new_list_teste)

def busca_Palavras(frases):
    todas_Palavras = []
    for(palavras, sentimento) in frases:
        todas_Palavras.extend(palavras)
    return todas_Palavras


palavras_treino = busca_Palavras(frases_com_Stem_treino)
palavras_teste = busca_Palavras(frases_com_Stem_teste)

print("Quantidade de palavras na base de Treino {}".format(pd.DataFrame(palavras_treino).count()))

def busca_frequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia_treino = busca_frequencia(palavras_treino)

frequencia_treino.most_common(20)

#Executamos também para a base de treino
frequencia_teste = busca_frequencia(palavras_teste)

def busca_palavras_unicas(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas_treino = busca_palavras_unicas(frequencia_treino)
palavras_unicas_teste = busca_palavras_unicas(frequencia_teste)

def extrator_palavras(documento):
    # Utilizando set() para associar a variavel doc com o parâmetro que esta chegando
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treino:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def extrator_palavras_teste(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_teste:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

base_completa_treino = nltk.classify.apply_features(extrator_palavras,frases_com_Stem_treino)

base_completa_teste = nltk.classify.apply_features(extrator_palavras_teste, frases_com_Stem_teste)

classificador = nltk.NaiveBayesClassifier.train(base_completa_treino)

print(classificador.labels())

print(classificador.show_most_informative_features(10))

print(nltk.classify.accuracy(classificador, base_completa_teste))

erros = []
for (frase, classe) in base_completa_teste:
    #print(frase)
    #print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))


esperado = []
previsto = []
for (frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print(matriz)

# Carregar os comentários do arquivo youtube_comments.csv
comments_df = pd.read_csv(YT_COMMENTS_PATH)
comments = comments_df['Comentário']

# Dicionários para armazenar o maior e menor valor de probabilidade para cada classe
maior_probabilidade = {'joy': (None, -1), 'sadness': (None, -1), 'anger': (None, -1), 
                       'fear': (None, -1), 'love': (None, -1), 'surprise': (None, -1)}

menor_probabilidade = {'joy': (None, 2), 'sadness': (None, 2), 'anger': (None, 2), 
                       'fear': (None, 2), 'love': (None, 2), 'surprise': (None, 2)}

# Pré-processar e classificar cada comentário
for comment in comments:
    # Verificar se o comentário é uma string
    if isinstance(comment, str):
        testeStemming = []
        stemmer = nltk.stem.RSLPStemmer()
        for palavras_treino in comment.split():
            comStem = [p for p in palavras_treino.split()]
            testeStemming.append(str(stemmer.stem(comStem[0])))

        novo = extrator_palavras(testeStemming)
        distribuicao = classificador.prob_classify(novo)
        
        print(f"Comentário: {comment}")
        for classe in distribuicao.samples():
            prob = distribuicao.prob(classe)
            print('%s: %f' % (classe, prob))
            
            # Atualizar maior e menor probabilidade
            if prob > maior_probabilidade[classe][1]:
                maior_probabilidade[classe] = (comment, prob)
            if prob < menor_probabilidade[classe][1]:
                menor_probabilidade[classe] = (comment, prob)
                
        print("\n")
        print("="*50)  # Adicionar linha de separação entre comentários
        print("\n")
    else:
        print(f"Comentário inválido: {comment}")

# Imprimir os comentários com maior e menor probabilidade para cada classe
print("\nComentários com maior probabilidade:")
for classe, (comment, prob) in maior_probabilidade.items():
    print(f"Classe: {classe}, Comentário: {comment}, Probabilidade: {prob}")
    print("\n")

print("\nComentários com menor probabilidade:")
for classe, (comment, prob) in menor_probabilidade.items():
    print(f"Classe: {classe}, Comentário: {comment}, Probabilidade: {prob}")
    print("\n")