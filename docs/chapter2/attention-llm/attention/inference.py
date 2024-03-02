import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn.functional as F


rcParams['font.family'] = 'SimHei'


def infer(model, inputs):
    with torch.no_grad():
        attn, outputs = model(inputs)
    prob = F.softmax(outputs, dim=-1)
    label = prob.argmax()
    return attn, label.item()


def plot_attention(attention_weights, source_words, target_words):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, interpolation='nearest', cmap='Blues')
    
    plt.xlabel('Source')
    plt.ylabel('Target')
    
    plt.xticks(np.arange(len(source_words)), source_words, rotation=45)
    plt.yticks(np.arange(len(target_words)), target_words)
    
    plt.colorbar()
    plt.tight_layout()
    plt.show()