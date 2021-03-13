import random

class Perceptron:
    
    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
        self.amostras = amostras
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.limiar = limiar
        self.n_amostras = len(amostras)
        self.n_atrib = len(amostras[0])
        self.pesos = []

    def treinar(self):
        for amostra in self.amostras:
            amostra.insert(0, -1)
            
        for i in range(self.n_atrib):
            self.pesos.append(random.random())

        self.pesos.insert(0, self.limiar)
        n_epocas = 0 #Contador de Epocas   

        while True:

            erro = False # Erro não Existe
            
            
            for i in range(self.n_amostras): 
                u = 0
                
                for j in range(self.n_atrib + 1):
                    u += self.pesos[j] * self.amostras[i][j]
                y = self.sinal(u) #Obtem a saida da rede

                # Verrifica a saida da rede é diferente da saida deseja.

                if y != self.saidas[i]:
                    #Calcular o erro
                    erro_aux = self.saidas[i] - y
                    # Faz ajustes dos pesos para cada elemento.
                    for j in range(self.n_atrib + 1):
                        self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
                    erro = True #O erro ainda é True    
            
            n_epocas += 1
            
            if not erro or n_epocas > self.epocas:
                break
    
    def teste(self, amostra):
        amostra.insert(0, -1)
        u = 0
        for i in range(self.n_atrib + 1 ):

            u += self.pesos[i] * amostra[i]
        y = self.sinal(u)
        print('Classe: %d' % y ) 



        
    def degrau(self, u):
        if u >= 0:
            return 1
        return 0

    def sinal(self, u):
        if u >= 0:
            return 1
        return -1

 

#entradas =[[0,0], [0,1], [1,0] , [1, 1]]
#saidas = [0, 1, 1, 1]

#rede = Perceptron(entradas, saidas, taxa_aprendizado=0.05)
#rede.treinar()
#rede.teste([0, 2])


# Outro Exemplo: 

entradas = [[0.1,0.4,0.7], [0.3,0.7,0.2], [0.6, 0.9, 0.8], [0.5, 0.7,0.1]]

saidas = [1, -1 , 1, 1]

rede = Perceptron(entradas, saidas)
rede.treinar()
rede.teste([0.1, 0.4,0.7])