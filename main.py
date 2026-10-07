import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


df_fake=pd.read_csv("DF/Fake.csv")
df_true=pd.read_csv("DF/True.csv")


#! Criando meu rotulo na qual Fake=1 e True=0
df_fake["label"]=1
df_true["label"]=0
df=pd.concat([df_fake,df_true],ignore_index=True)#!Juntei os dois
df["texto_completo"]=df["title"]+" "+df["text"]
df=df[["label","texto_completo","subject"]]

df_politica = df[df['subject'].isin(['politics', 'politicsNews'])]#!Filtro apenas para politicas
df_news=df[df["subject"].isin(['News','worldnews'])]

#! Verificar quais palavras mais aparece
def verificar_pesos():
    nomes=vetorizador.get_feature_names_out()
    coeficientes= modelo_svm.coef_[0]
    indices_fake=np.argsort(coeficientes)[-20:]
    print("=== Top 20 Palavras/Indicadores mais associados a FAKE NEWS ===")
    for idx in reversed(indices_fake):
        print(f"{nomes_palavras[idx]}: {coeficientes[idx]:.4f}")