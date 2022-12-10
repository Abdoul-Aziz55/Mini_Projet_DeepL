import sys
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import time



#lecture du fichier et conversion en objet datetime les dates qui étaient en format string
df = pd.read_csv("Data/stock-quotidien-stockages-gaz.csv", delimiter=";", parse_dates=["date"])

df = df.sort_values(by='date') #tri en fonction de la date
df = df[df["pits"] == "Centre"] # recuperation d'un seul "pits" qu'on va suivre.

df = df[df['date'] > pd.to_datetime("2019-12-31")] #recuperation d'une partie des données
df = df[df['date'] < pd.to_datetime("2021-1-1")] #recuperation d'une partie des données
df = df.iloc[:, :2].set_index('date') #on supprime les features qui nous interessent pas

test_size = int((df.shape[0]*20)/100) # 20% des données utilisées pour tester le modele
train_data = df[:-test_size]
test_data = df[-test_size:]

#print(train_data.shape)
#print(test_data.shape)

#normalisation des donnees
scaler = MinMaxScaler()
scaler = scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

def fenetre_glissante(data, longueur_sequence):
    xs = []
    ys = []
    for i in range(len(data) - longueur_sequence - 1):
        xs.append(data[i:(i+longueur_sequence)]) #les valeurs utilisees pour faire la prediction
        ys.append(data[i+longueur_sequence]) #valeur à predire

    return np.array(xs), np.array(ys)

longueur_sequence = 10 # on utilise 10 jours pour predire le 11eme

x_train, y_train = fenetre_glissante(train_data, longueur_sequence)
x_test, y_test = fenetre_glissante(test_data, longueur_sequence)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

x_test = torch.from_numpy(x_test).float() #meme shape que x_train
y_test = torch.from_numpy(y_test).float() #meme shape que y_train

print(x_train.shape, y_train.shape)
# shape x_train: (nombre de sequence, longueur sequence, 1) 1: nombre de features
# shape y_train: (nombre de sequence, 1) nombre de sequence: chaque valeur correspond à la vraie valeur qu'on veut predire pour chaque valeur de x_train
# 1: car on veut predire une seule valeur en fonction longueur sequence valeurs
print(x_test.shape, y_test.shape)
#meme shapes que x_train et y_train

trainds = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
criterion = nn.MSELoss()

# "------------"
class PredicteurStockQuotidien(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers=2):
        super(PredicteurStockQuotidien, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.5
        )

        self.linear = nn.Linear(
            in_features=hidden_dim, #on prend en entree le nombre de valeurs sortant du LSTM
            out_features=1 #on predit une seule valeur
        )

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim),
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim)
        )

    def forward(self, input):
        lstm_out, _ = self.lstm(
            input.view(len(input), self.seq_length, -1),
            self.hidden
        )
        y_pred = self.linear(
            lstm_out.view(self.seq_length, len(input), self.hidden_dim)[-1]
        )

        return y_pred

def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = criterion(haty,goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss

def train(mod, nepochs, learning_rate):
    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    testLossVector = np.zeros(nepochs)
    trainLossVector = np.zeros(nepochs)
    for epoch in range(nepochs):
        mod.reset_hidden_state()
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = criterion(haty, goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        testLossVector[epoch] = testloss
        trainLossVector[epoch] = totloss
        print(f'Epoch {epoch} train_loss: {totloss} test_loss: {testloss}')
    print(f'Fin Epoch {epoch} train_loss: {totloss} test_loss: {testloss}', file=sys.stderr)
    return mod.eval(), trainLossVector, testLossVector
def train_CNN(mod, nepochs, learning_rate):
    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    testLossVector = np.zeros(nepochs)
    trainLossVector = np.zeros(nepochs)
    for epoch in range(nepochs):
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = criterion(haty, goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        testLossVector[epoch] = testloss
        trainLossVector[epoch] = totloss
        print(f'Epoch {epoch} train_loss: {totloss} test_loss: {testloss}')
    print(f'Fin Epoch {epoch} train_loss: {totloss} test_loss: {testloss}', file=sys.stderr)
    return mod.eval(), trainLossVector, testLossVector

def using_CNN():
    layer= (torch.nn.Conv1d(10,1,1), torch.nn.ReLU())
    mod = torch.nn.Sequential(*layer)
    start = time()
    mod, train_hist, test_hist = train_CNN(mod, n_epochs, learningRate)
    print('training time', time()-start)  
    plt.plot(train_hist, label='train loss')
    plt.plot(test_hist, label='test loss')
    plt.legend(loc="upper right")
    plt.title('using_cnn')
    plt.show()

def using_LSTM():
    
    model = PredicteurStockQuotidien(input_dim, hidden_dim, longueur_sequence, n_layers)
    start = time()

    model, train_hist, test_hist = train(model, n_epochs, learningRate)
    print('training time', time()-start)

    plt.plot(train_hist, label='train loss')
    plt.plot(test_hist, label='test loss')
    plt.legend(loc="upper right")
    plt.title('using_LSTM')
    plt.show()

input_dim = 1
hidden_dim = 10
n_layers = 2
n_epochs = 20
learningRate = 0.001

print('using CNN')
using_CNN()
print('using LSTM')
using_LSTM()