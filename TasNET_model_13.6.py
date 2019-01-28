import torch
import torch.nn as nn
import numpy as np

class TasNET(nn.Module):
    def __init__(self,batch_size,samples=40,features=500, num_layers=2, hidden_size=500, dropout=0,
                 bidirectional=True):
        super(TasNET, self).__init__()
        self.features = features
        self.batch = batch_size
        self.samples = samples        
        self.conv1 = nn.Linear(samples,features,bias=False)
        self.conv2 = nn.Linear(samples,features,bias=False)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.LayerNorm = nn.LayerNorm(features,0)
        self.LSTM12 = nn.LSTM(features, hidden_size,num_layers,batch_first=True, 
                        dropout=dropout,bidirectional=bidirectional)
        self.LSTM34 = nn.LSTM(2*features, hidden_size,num_layers,batch_first=True, 
                        dropout=dropout,bidirectional=bidirectional)
        self.Linear = nn.Linear(2*features,2*features)
        self.Softmax = nn.Softmax(-1)
        self.reconv = nn.Linear(features,samples,bias=False)
    
    def forward(self, x):
        #apply overlap
        x = x.view(self.batch, -1)
        x = torch.cat((x,x[:,-int(self.samples/2):]),1)
        x = x.unfold(1,self.samples,int(self.samples/2))
        # print(x.shape)

        #Normalize before the Tasnet
        fac = x.std(2)   #(128,100,40)
        fac = fac.view(self.batch,-1,1)  #(128,100,1)
        x = x/fac    #(128,100,40)

        #The encoder part in the paper 
        x_conv1 = self.conv1(x)        #(128,100,500)
        x_conv2 = self.conv2(x)        #(128,100,500)
        x_non1 = self.ReLU(x_conv1)        #(128,100,500)
        x_non2 = self.Sigmoid(x_conv2)     #(128,100,500)
        w = x_non1 * x_non2              #(128,100,500)   

        #Layer norm before LSTM
        x = self.LayerNorm(w)    #(128,100,500)

        #LSTM layers and skip connection
        x_LSTM2, _ = self.LSTM12(x,None)
        x_LSTM4, _ = self.LSTM34(x_LSTM2,None)
        x = torch.add(x_LSTM2, x_LSTM4)

        #Linear and softmax layers after LSTM
        x = self.Linear(x)          #(128,100,500*2)
        x = x.view(self.batch,-1,2)  #(128,100*500,2)
        m = self.Softmax(x)  #(128,100*500,2)
        m1 = m[:,:,0]         #(128,100*500,1)
        m2 = m[:,:,1]             
        m1 = m1.view(self.batch,-1,self.features)  #(128,100,500)
        m2 = m2.view(self.batch,-1,self.features)
        x1 = w*m1
        x2 = w*m2

        #Decoder layer in the paper 
        x1 = self.reconv(x1)        #(128,100,40) 
        x2 = self.reconv(x2)        #(128,100,40)

        #Reverse the effect of normalization before Tasnet
        x1 = x1*fac
        x2 = x2*fac

        #Reconstruct the waves
        x1 = x1.narrow(2,0,int(self.samples/2))
        x2 = x2.narrow(2,0,int(self.samples/2))
        x1 = x1.contiguous().view(self.batch,-1)           #(128,4000)
        x2 = x2.contiguous().view(self.batch,-1)           #(128,4000)

        return x1, x2

def test_model():
    x = torch.FloatTensor(torch.randn(128,100,40))
    # fac = torch.std(x,2)
    # fac = fac.view(128,100,1)
    # print(x[1,1,:])
    # print(fac.shape)
    # x = x/fac
    # print(x.std(2))
    # x = x*fac
    # print(x[1,1,:])
    Tas = TasNET(128)
    output1, output2=Tas(x)
    #print(output)
    print(output1.size())
    print(output2.size())
    print(Tas)


if __name__ == '__main__':
    test_model()

