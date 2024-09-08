import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.VGGTextCNN import VGGTextCNN

data = pd.read_csv("data/processed_training_data.csv")

# 创建模型实例
num_classes = len(data['label'].unique())
model = VGGTextCNN(num_classes)
model = model.to(device)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from save_load_checkpoint import load_checkpoint

# 选择TextCNN做用户请求的文本分类
model_path = 'checkpoints/TextCNN/TextCNN.pth'
optimizer_path = 'checkpoints/TextCNN/optimizer.pth'

# 加载模型和优化器状态
model, optimizer, start_epoch = load_checkpoint(model, optimizer, model_path, optimizer_path)

# 加载训练好的模型
model = model.to(device)
model.eval()

import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

from openai import OpenAI
api_key = ""
api_base = ""
client = OpenAI(api_key=api_key, base_url=api_base)
dialogue_model = "gpt-4o-mini" # "gpt-3.5-turbo"
embed_model = 'text-embedding-3-large' # 嵌入用个贵的

import tiktoken
# 初始化encoding
encoding = tiktoken.get_encoding("cl100k_base")
# 先看看所有数据中的token数的分布
encoding = tiktoken.encoding_for_model(embed_model) # 实例化

import pickle
from collections import defaultdict

# 从文件加载embedding_cache
def load_embedding_cache(filename='embedding_cache.pkl'):
    with open(filename, 'rb') as f:
        cache_dict = pickle.load(f)
    # 重新构建defaultdict并设置默认值
    cache = defaultdict(lambda: None, cache_dict)
    return cache
    
# 加载embedding_cache
embedding_cache = load_embedding_cache()

max_len = 24

import re

# 加载label_encoder
label_encoder = {label: idx for idx, label in enumerate(data['label'].unique())}
idx_to_label = {idx: label for label, idx in label_encoder.items()}

# 数据预处理函数
def preprocess_user_input(text, max_len):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = encoding.encode(text)
    letters = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in tokens]
    embeddings = [get_single_letter_embedding(letter) for letter in letters]
    embeddings = torch.tensor(embeddings)

    if embeddings.shape[0] < max_len:
        padding = torch.zeros((max_len - embeddings.shape[0], embeddings.shape[1]))
        embeddings = torch.cat((embeddings, padding), dim=0)
    else:
        embeddings = embeddings[:max_len]

    return embeddings.unsqueeze(0).unsqueeze(1)  # 增加batch维度，然后增加通道数

# 获取单个字母嵌入的函数
def get_single_letter_embedding(letter):
    if embedding_cache[letter] is not None:
        return embedding_cache[letter]
    
    response = client.embeddings.create(input=letter, model=embed_model)
    embedding = response.data[0].embedding
    embedding_cache[letter] = embedding
    return embedding

# 预测函数
def predict(text):
    with torch.no_grad():
        embeddings = preprocess_user_input(text, max_len).to(device)
        outputs = model(embeddings)
        probabilities = F.softmax(outputs, dim=1).squeeze()
        # print(probabilities)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = idx_to_label[predicted_idx]
        predicted_prob = probabilities[predicted_idx].item()
        
    return predicted_label, predicted_prob

def request_processor(request):
    # user_request = "what is the positional encoding?"
    predicted_label, predicted_prob = predict(request)
    # print(f"Predicted label: {predicted_label}, Probability: {predicted_prob}")
    return predicted_label,predicted_prob




