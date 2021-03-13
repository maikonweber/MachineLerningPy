#Implentação de Knn
import math


#Lista de Amostra

amostras = []


with open('dataset.data', "r") as f:
    for linhas in f.readlines():
        atrib = linhas.replace("\n", '').split(',')
        amostras.append([int(atrib[0]), int(atrib[1]), int(atrib[2]), int(atrib[3])])



def info_dataset( amostras , verbose=True ):
    if verbose:
            print("Total de Amostra: %d" % len(amostras))
    rotulo1, rotulo2 = 0 , 0
    for amostra in amostras:
        if amostra[-1] == 1:
            rotulo1 +=1

        else:
            rotulo2 += 1
    if verbose:
        print("Total Rotulo 1: %d" % rotulo1)
        print("Total Rotulo 2: %d" % rotulo2)
    
    return ([len(amostras), rotulo1, rotulo2])


p = 0.6


_, rotulo1, rotulo2 = info_dataset(amostras, verbose=False)

treinamento, teste = [], []

max_rotulo1, max_rotulo2 =int(p * rotulo1) , int(p * rotulo2)

total_rotulo1, total_rotulo2 = 0 , 0

for amostra in amostras:
    if (total_rotulo1 + total_rotulo2) < (max_rotulo1 + max_rotulo2):

        treinamento.append(amostra)

        if amostra[-1] == 1 and total_rotulo1 < max_rotulo1:
            total_rotulo1 += 1
        else:
            total_rotulo2 += 1
    else:
        teste.append(amostra)


info_dataset(treinamento)





def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range (dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)

    return math.sqrt(soma)

# Teste da Distância Euclida

#v1 = [1, 2, 3]
#v2 = [2, 1, 3]

#print(dist_euclidiana(v1, v2))
# Resultado 1.41



def knn(treinamento, nova_amostra, K):
    dists, tam_treino = {}, len(treinamento)
    # Calcular a ditância euclidiana da amostra para todos os exmplos de conjusnto de treinamento

    for i in range (tam_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d 


       # obtém as chaves (indices) dos k-vizinhos mais próximos

#d = {1:2.34, 2:3.45, 3:0.45, 4:9.8}

#print(sorted(d, key=d.get)[:3])

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    # Votação Majoritária 

    qtd_rotulo1, qtd_rotulo2 = 0, 0
    for indice in k_vizinhos: 
        if treinamento[indice][-1] == 1:
            qtd_rotulo1 += 1
        else:
            qtd_rotulo2 += 1 
        if qtd_rotulo1 > qtd_rotulo2:
            return 1
        else:
            return 2



acertos, K = 0, 15

for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:    
        acertos += 1

print('Total de treinamento %d' %len(treinamento))
print('Total de teste %d' % len(teste))
print('Total de Acertos %d' % acertos)
print('Porcentagem de Acertos : %.2f%%' % (100 * acertos / len(teste)))

