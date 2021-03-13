#Impplementação do Knn com Sklearn 


from sklearn.neighbors import KNeighborsClassifier

entradas, saidas = [], [] # Array de Entradas e Saidas

with open('dataset.data', 'r') as f: #Abrir o Dataset em modo leitura como F
    for linha in f.readlines():  #Para variaveis Linha, Leia F linha
        atrib = linha.replace('\n', '').split(',') #Retirar o \n separar por '' e dividir por ,
        entradas.append([int(atrib[0]), int(atrib[2])]) # Atribui o na valores na lista entrada
        saidas.append(int(atrib[3])) #Atribui o valores na lista saidas.



p = 0.6 #Porcentagem de dados para treinamento.

limite = int(p * len(entradas))

knn = KNeighborsClassifier(n_neighbors=15) #Classificador do Sklearn "recebe atributos" - Verificar na Documentação

knn.fit(entradas[:limite], saidas[:limite]) #Metodos para Iniciar o Treinamento.

labels = knn.predict(entradas[limite:]) # Fazer previsão para um outra parte de dados

acertos, indice_label = 0 , 0

for i in range(limite, len(entradas)): #Testar as amostras dos conjustos dos Testes.
    if labels[indice_label] == saidas[i]:
        acertos += 1
    indice_label += 1

print("Total de Treinamentos: %d" % limite )
print("Total de testes: %d" % (len(entradas) - limite ))
print("Total de Acertos: %d "  % acertos )
print('Porcentagem de Acertos : %.2f ' % (100 * acertos / (len(entradas) - limite )))