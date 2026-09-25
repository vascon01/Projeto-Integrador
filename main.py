import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df_fake=pd.read_csv("/home/gabriel/Obsidian Vault/​🗂️​ FACULDADE/🗂️ 4° Semestre/Projeto Integrador/Codigo/DF/Fake.csv")
df_true=pd.read_csv("/home/gabriel/Obsidian Vault/​🗂️​ FACULDADE/🗂️ 4° Semestre/Projeto Integrador/Codigo/DF/True.csv")


#! Criando meu rotulo na qual Fake=1 e True=0
df_fake["label"]=1
df_true["label"]=0
df=pd.concat([df_fake,df_true],ignore_index=True)#!Juntei os dois
df["texto_completo"]=df["title"]+" "+df["text"]
df=df[["label","texto_completo","subject"]]

df_politica = df[df['subject'].isin(['politics', 'politicsNews'])]#!Filtro apenas para politicas

#!Treinamento do primeiro modelo Naive Bayes
# !---------------------------------------------------------

X=df_politica["texto_completo"]
Y=df_politica["label"]

#Divisao de treino e teste 80 treino 20 teste 
X_treino,X_teste,Y_treino,Y_teste=train_test_split(X,Y,test_size=0.20,random_state=4, stratify=Y)

# 6. Vetorização de texto usando TF-IDF
vetorizador = TfidfVectorizer(max_features=10000)
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)

# 7. Treinamento do modelo Naive Bayes
modelo_nb = MultinomialNB()
modelo_nb.fit(X_treino_vetorizado, Y_treino)

# 8. Predições e Resultados
predicoes = modelo_nb.predict(X_teste_vetorizado)

print('=== Acurácia no Subconjunto de Política ===')
print(f'{accuracy_score(Y_teste, predicoes) * 100:.2f}%\n')

print('=== Relatório de Classificação ===')
print(classification_report(Y_teste, predicoes, target_names=['True', 'Fake']))