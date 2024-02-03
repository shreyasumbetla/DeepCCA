import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # print("x in mlp = ", x.shape)
            x = layer(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size,device):
        super(SelfAttention, self).__init__()

        self.device = device
        self.query_projection = nn.Linear(input_size, attention_size)#.to(device)
        self.key_projection = nn.Linear(input_size, attention_size)#.to(device)
        self.value_projection = nn.Linear(input_size, attention_size)#.to(device)

        self.scale_factor = torch.sqrt(torch.FloatTensor([attention_size])).to(device)

    def forward(self, x):
        # print("x = ",x.shape)
        # Project inputs to query, key, and value
        query = self.query_projection(x).to(self.device)
        key = self.key_projection(x).to(self.device)
        value = self.value_projection(x).to(self.device)
        self.scale_factor.to(self.device)
        # print("key = ",key.shape)
        # Compute scaled dot-product attention scores
        # print("query d = ", query.get_device())
        # print("key d = ", key.get_device())
        # print("scale d = ", self.scale_factor.get_device())
        key = key.to(query.device)
        self.scale_factor = self.scale_factor.to(query.device)
        scores = (torch.matmul(query, key.transpose(-2,-1)) / self.scale_factor)#.to(self.device)
        # print("scores = ", scores.shape )
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)
        # print("attented values = ", attended_values.shape)
        return attended_values
        
# class AttentionModel(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(AttentionModel, self).__init__()

#         # Define layers for the first input
#         self.fc1_input1 = nn.Linear(input_size, hidden_size)
#         self.fc2_input1 = nn.Linear(hidden_size, hidden_size)

#         # Define layers for the second input
#         self.fc1_input2 = nn.Linear(input_size, hidden_size)
#         self.fc2_input2 = nn.Linear(hidden_size, hidden_size)

#         # Attention mechanism parameters
#         self.attention_weights = nn.Parameter(torch.rand(hidden_size))

#         # Fully connected layer for the final output
#         self.fc_final = nn.Linear(hidden_size, 1)

#     def forward(self, input1, input2):
#         # Process the first input
#         x1 = F.relu(self.fc1_input1(input1))
#         x1 = F.relu(self.fc2_input1(x1))

#         # Process the second input
#         x2 = F.relu(self.fc1_input2(input2))
#         x2 = F.relu(self.fc2_input2(x2))

#         # Apply attention mechanism
#         attention_scores = F.softmax(self.attention_weights, dim=0)
#         attended_input1 = attention_scores * x1
#         attended_input2 = attention_scores * x2

#         # Calculate MSE loss
#         mse_loss = F.mse_loss(attended_input1, attended_input2)

#         # Final output
#         output = self.fc_final(attended_input1)

#         return output, mse_loss    

class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        
        super(DeepCCA, self).__init__()

        ######## Simple Attention Block
        # """
        self.attention_size = 1024
        self.attention1 = SelfAttention(input_size1,self.attention_size, device).to(device)
        self.attention2 = SelfAttention(input_size2,self.attention_size, device).to(device)
        self.model1 = MlpNet(layer_sizes1, self.attention_size).double()
        self.model2 = MlpNet(layer_sizes2, self.attention_size).double()
        # """
        self.device = device

        ######## Transformer Encoder Layer 
        # self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=784, nhead=1,batch_first=True,device=device)
        # self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=1)

        # self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=784, nhead=1,batch_first=True,device=device)
        # self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)

        # self.dim_encoder = 784 #self.transformer_encoder.encoder_layer.dim_feedforward
        # self.model1 = MlpNet(layer_sizes1, self.dim_encoder).double()
        # self.model2 = MlpNet(layer_sizes2, self.dim_encoder).double()


        ########## Original Simple MLP 
        # self.model1 = MlpNet(layer_sizes1, input_size1).double()
        # self.model2 = MlpNet(layer_sizes2, input_size2).double()

        # if loss_type == 'MSE':
        #     print("loss is MSE!!!!!!")
        #     self.loss = nn.MSELoss()
        # else:
        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        #self.loss =  F.mse_loss(x2, attended_values)

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        ######## Simple Attention Block
        output_att1 = self.attention1(x1)
        output_att2 = self.attention2(x2)
        output1 = self.model1(output_att1)
        output2 = self.model2(output_att2)

        ######## Transformer Encoder Layer + MLP
        # output_enc1 = self.transformer_encoder1(x1)
        # output_enc2 = self.transformer_encoder2(x2)
        # output1 = self.model1(output_enc1)
        # output2 = self.model2(output_enc2)


        ########## Original Simple MLP 
        # output1 = self.model1(x1)
        # output2 = self.model2(x2)

        return output1, output2
