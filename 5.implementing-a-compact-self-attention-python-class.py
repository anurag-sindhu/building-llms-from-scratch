from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C


    
#A The second input element
#B The input embedding size, d=3
#C The output embedding size, d_out=2

print("x_2", x_2)
print("d_in", d_in)
print("d_out", d_out)


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2, key_2, value_2)

#Next, we compute the query, key, and value vectors as shown earlier
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

#We can obtain all keys and values via matrix multiplication:
keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)


# First, let's compute the attention score
keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)
print ("attn_score_22", attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print("attn_scores_2", attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print( "attn_weights_2", attn_weights_2)

#We now compute the context vector as a weighted sum over the value
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("sa_v1(inputs)", sa_v1(inputs))