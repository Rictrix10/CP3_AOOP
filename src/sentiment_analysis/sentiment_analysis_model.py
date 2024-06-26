import pandas as pd
import numpy as np
import nltk
import pandas as pd
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

seed = 42

new_dataset = exemplo_base

print('Tamanho da base balanceada {}'.format(new_dataset.shape[0]))

print(new_dataset.Emotion.value_counts())

new_dataset.sample(n=20)
print(new_dataset.sample(n=20))

lista_Stop = nltk.corpus.stopwords.words('english')
print(np.transpose(lista_Stop))

def removeStopWords(texto):
    frases = []
    for (palavras, sentimento) in texto:
        
        semStop = [p for p in palavras.split() if p.lower() not in lista_Stop]
        
        frases.append((semStop, sentimento))
    return frases

def aplica_Stemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    
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

print(frequencia_treino.most_common(20))

frequencia_teste = busca_frequencia(palavras_teste)

def busca_palavras_unicas(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas_treino = busca_palavras_unicas(frequencia_treino)
palavras_unicas_teste = busca_palavras_unicas(frequencia_teste)

def extrator_palavras(documento):
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

comments_df = pd.read_csv(YT_COMMENTS_PATH)
comments = comments_df['Comentário']


maior_probabilidade = {'joy': (None, -1), 'sadness': (None, -1), 'anger': (None, -1), 
                       'fear': (None, -1), 'love': (None, -1), 'surprise': (None, -1)}

menor_probabilidade = {'joy': (None, 2), 'sadness': (None, 2), 'anger': (None, 2), 
                       'fear': (None, 2), 'love': (None, 2), 'surprise': (None, 2)}

for comment in comments:
    
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
            
            
            if prob > maior_probabilidade[classe][1]:
                maior_probabilidade[classe] = (comment, prob)
            if prob < menor_probabilidade[classe][1]:
                menor_probabilidade[classe] = (comment, prob)
                
        print("\n")
        print("="*50)  
        print("\n")
    else:
        print(f"Comentário inválido: {comment}")


print("\nComentários com maior probabilidade:")
for classe, (comment, prob) in maior_probabilidade.items():
    print(f"Classe: {classe}, Comentário: {comment}, Probabilidade: {prob}")
    print("\n")

print("\nComentários com menor probabilidade:")
for classe, (comment, prob) in menor_probabilidade.items():
    print(f"Classe: {classe}, Comentário: {comment}, Probabilidade: {prob}")
    print("\n")