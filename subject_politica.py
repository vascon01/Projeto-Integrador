from main import *

#!Treinamento do primeiro modelo Naive Bayes com os dados do df_politica
# !---------------------------------------------------------

X=df_politica["texto_completo"]
Y=df_politica["label"]

#Divisao de treino e teste 80 treino 20 teste 
X_treino,X_teste,Y_treino,Y_teste=train_test_split(X,Y,test_size=0.20,random_state=4, stratify=Y)

#Vetorização de texto usando TF-IDF
vetorizador = TfidfVectorizer(max_features=10000)
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)


#! MODELO 1: Naive Bayes

modelo_nb = MultinomialNB()
modelo_nb.fit(X_treino_vetorizado, Y_treino)
predicoes_nb = modelo_nb.predict(X_teste_vetorizado)

print("=== Naive Bayes ===")
print(f"Acurácia: {accuracy_score(Y_teste, predicoes_nb) * 100:.2f}%\n")
print(classification_report(Y_teste, predicoes_nb, target_names=['True', 'Fake']))


#! MODELO 2: Linear SVC

modelo_svm = LinearSVC(dual=False,random_state=42)
modelo_svm.fit(X_treino_vetorizado, Y_treino)
predicoes_svm = modelo_svm.predict(X_teste_vetorizado)  # <-- Predição própria do SVM!

print("=== Linear SVC ===")
print(f"Acurácia: {accuracy_score(Y_teste, predicoes_svm) * 100:.2f}%\n")
print(classification_report(Y_teste, predicoes_svm, target_names=['True', 'Fake']))


#!MODELO 3: SGDClassifier (Novo)

modelo_sgd = SGDClassifier(loss='hinge', random_state=42)
modelo_sgd.fit(X_treino_vetorizado, Y_treino)
predicoes_sgd = modelo_sgd.predict(X_teste_vetorizado)

print('=== SGDClassifier ===')
print(f'Acurácia: {accuracy_score(Y_teste, predicoes_sgd) * 100:.2f}%\n')
print('=== Relatório de Classificação (SGDClassifier) ===')
print(classification_report(Y_teste, predicoes_sgd, target_names=['True', 'Fake']))
