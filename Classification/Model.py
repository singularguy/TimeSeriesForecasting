import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. LSTM + Attention
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 2. GRU + Attention
class GRUWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 3. CNN + LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        return output

# 4. CNN + GRU
class CNN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 5. LSTM + CNN
class LSTM_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 取最后一个时间步的输出
        output = self.fc(x[:, -1, :])
        return output

# 6. LSTM + Bi-directional GRU
class LSTM_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(lstm_out)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 7. GRU + Bi-directional LSTM
class GRU_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(bilstm_out[:, -1, :])
        return output

# 8. LSTM + CNN + Attention
class LSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 9. GRU + CNN + Attention
class GRU_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 10. BiLSTM + Multihead Attention
class BiLSTM_MultiheadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4):
        super(BiLSTM_MultiheadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Multihead Attention层用于加权序列中的重要信息
        self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过Multihead Attention层处理输入序列
        attn_output, _ = self.multihead_attn(bilstm_out, bilstm_out, bilstm_out)
        # 取最后一个时间步的输出
        output = self.fc(attn_output[:, -1, :])
        return output

# 11. GRU + Transformer Encoder
class GRU_TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(GRU_TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 12. CNN + Transformer
class CNN_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 13. LSTM + Transformer
class LSTM_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(lstm_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 14. CNN + RNN (LSTM/GRU)
class CNN_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, rnn_type='LSTM'):
        super(CNN_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(x)
        # 取最后一个时间步的输出
        output = self.fc(rnn_out[:, -1, :])
        return output

# 15. GRU + RNN (LSTM/GRU)
class GRU_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, rnn_type='LSTM'):
        super(GRU_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(rnn_out[:, -1, :])
        return output

# 16. LSTM + GRU + Attention
class LSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 17. LSTM + Dense + Attention
class LSTM_Dense_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Dense_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(lstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(dense_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * dense_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 18. GRU + Dense + Attention
class GRU_Dense_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU_Dense_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(dense_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * dense_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 19. Transformer + Multihead Attention
class Transformer_MultiheadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(Transformer_MultiheadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Multihead Attention层用于加权序列中的重要信息
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))
        # 通过Multihead Attention层处理输入序列
        attn_output, _ = self.multihead_attn(transformer_out, transformer_out, transformer_out)
        # 取最后一个时间步的输出
        output = self.fc(attn_output[-1, :, :])
        return output

# 20. CNN + RNN (LSTM/GRU) + Attention
class CNN_RNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, rnn_type='LSTM'):
        super(CNN_RNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(rnn_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * rnn_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 21. BiLSTM + GRU
class BiLSTM_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 22. Transformer + CNN
class Transformer_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(Transformer_CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))
        # 通过CNN层提取局部特征
        x = transformer_out.permute(1, 2, 0)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 取最后一个时间步的输出
        output = self.fc(x[-1, :, :])
        return output

# 23. GRU + CNN + BiLSTM
class GRU_CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 取最后一个时间步的输出
        output = self.fc(bilstm_out[:, -1, :])
        return output

# 24. LSTM + CNN + GRU
class LSTM_CNN_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 25. LSTM + RNN (LSTM/GRU) + Dense
class LSTM_RNN_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, rnn_type='LSTM'):
        super(LSTM_RNN_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(lstm_out)
        # 通过Dense层进行特征映射
        dense_out = self.dense(rnn_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 26. GRU + CNN + Dense
class GRU_CNN_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Dense层用于特征映射
        self.dense = nn.Linear(filters, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Dense层进行特征映射
        dense_out = self.dense(x)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 27. CNN + Transformer + Attention
class CNN_Transformer_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_Transformer_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(transformer_out), dim=0)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * transformer_out, dim=0)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 28. LSTM + CNN + Transformer
class LSTM_CNN_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(LSTM_CNN_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 29. BiLSTM + CNN + Attention
class BiLSTM_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(BiLSTM_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim * 2, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过CNN层提取局部特征
        x = bilstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 30. GRU + Dense + CNN
class GRU_Dense_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_Dense_CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 通过CNN层提取局部特征
        x = dense_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 取最后一个时间步的输出
        output = self.fc(x[:, -1, :])
        return output

# 31. LSTM + CNN + GRU + Attention
class LSTM_CNN_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 32. LSTM + Transformer + Dense
class LSTM_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(lstm_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 33. BiLSTM + CNN + Transformer
class BiLSTM_CNN_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(BiLSTM_CNN_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim * 2, filters, kernel_size, padding=(kernel_size-1)//2)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过CNN层提取局部特征
        x = bilstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 34. GRU + Transformer + Dense
class GRU_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(GRU_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 35. LSTM + Bi-directional GRU + Attention
class LSTM_BiGRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_BiGRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(lstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bigru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bigru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 36. CNN + LSTM + BiLSTM
class CNN_LSTM_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_LSTM_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(lstm_out)
        # 取最后一个时间步的输出
        output = self.fc(bilstm_out[:, -1, :])
        return output

# 37. CNN + LSTM + Bi-directional GRU
class CNN_LSTM_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_LSTM_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(lstm_out)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 38. LSTM + CNN + Bi-directional GRU
class LSTM_CNN_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(x)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 39. LSTM + RNN (LSTM/GRU) + Dense
class LSTM_RNN_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, rnn_type='LSTM'):
        super(LSTM_RNN_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(lstm_out)
        # 通过Dense层进行特征映射
        dense_out = self.dense(rnn_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 40. GRU + Transformer + Dense
class GRU_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(GRU_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 41. LSTM + GRU + Transformer
class LSTM_GRU_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_GRU_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 42. LSTM + CNN + RNN (LSTM/GRU)
class LSTM_CNN_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, rnn_type='LSTM'):
        super(LSTM_CNN_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(x)
        # 取最后一个时间步的输出
        output = self.fc(rnn_out[:, -1, :])
        return output

# 43. LSTM + Bi-directional LSTM + Transformer
class LSTM_BiLSTM_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_BiLSTM_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(lstm_out)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(bilstm_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 44. CNN + Attention + Transformer
class CNN_Attention_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_Attention_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(context_vector.unsqueeze(0).permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 45. LSTM + CNN + RNN + Attention
class LSTM_CNN_RNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, rnn_type='LSTM'):
        super(LSTM_CNN_RNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(rnn_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * rnn_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 46. LSTM + CNN + Bi-directional LSTM + Attention
class LSTM_CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bilstm_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 47. GRU + CNN + Bi-directional LSTM + Attention
class GRU_CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bilstm_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 48. CNN + BiLSTM + GRU + Attention
class CNN_BiLSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_BiLSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 49. Transformer + Bi-directional GRU
class Transformer_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(Transformer_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 50. LSTM + Dense + CNN + Attention
class LSTM_Dense_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_Dense_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(lstm_out)
        # 通过CNN层提取局部特征
        x = dense_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 51. GRU + CNN + Bi-directional LSTM + Attention
class GRU_CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bilstm_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 52. BiLSTM + CNN + GRU + Dense
class BiLSTM_CNN_GRU_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(BiLSTM_CNN_GRU_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim * 2, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过CNN层提取局部特征
        x = bilstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 53. LSTM + Transformer + GRU
class LSTM_Transformer_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_Transformer_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(lstm_out.permute(1, 0, 2))
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 54. CNN + RNN (LSTM/GRU) + Transformer
class CNN_RNN_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, rnn_type='LSTM', num_heads=4, dim_feedforward=128):
        super(CNN_RNN_Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # RNN层用于处理序列数据,捕捉时间序列中的依赖关系
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("RNN type must be 'LSTM' or 'GRU'")
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过RNN层处理输入序列
        rnn_out, _ = self.rnn(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(rnn_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(transformer_out[-1, :, :])
        return output

# 55. GRU + CNN + Dense
class GRU_CNN_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Dense层用于特征映射
        self.dense = nn.Linear(filters, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Dense层进行特征映射
        dense_out = self.dense(x)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 56. BiLSTM + CNN + Dense + Attention
class BiLSTM_CNN_Dense_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(BiLSTM_CNN_Dense_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim * 2, filters, kernel_size, padding=(kernel_size-1)//2)
        # Dense层用于特征映射
        self.dense = nn.Linear(filters, hidden_dim)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过CNN层提取局部特征
        x = bilstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Dense层进行特征映射
        dense_out = self.dense(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(dense_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * dense_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 57. LSTM + Transformer + GRU
class LSTM_Transformer_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_Transformer_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(lstm_out.permute(1, 0, 2))
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 58. CNN + GRU + Attention + Dense
class CNN_GRU_Attention_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_GRU_Attention_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过Dense层进行特征映射
        dense_out = self.dense(context_vector)
        # 通过全连接层输出最终结果
        output = self.fc(dense_out)
        return output

# 59. Bi-directional LSTM + GRU + Attention
class BiLSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 60. LSTM + CNN + GRU + Dense
class LSTM_CNN_GRU_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_GRU_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 61. LSTM + GRU + CNN + Attention
class LSTM_GRU_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_GRU_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 62. LSTM + GRU + CNN + Bi-directional GRU
class LSTM_GRU_CNN_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_GRU_CNN_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(x)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 63. GRU + Transformer + Dense
class GRU_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(GRU_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 64. LSTM + CNN + Bi-directional GRU + Dense
class LSTM_CNN_BiGRU_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_BiGRU_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim * 2, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(bigru_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 65. CNN + GRU + Transformer + Dense
class CNN_GRU_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_GRU_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 66. Bi-directional LSTM + GRU + Attention
class BiLSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 67. LSTM + GRU + CNN + Attention
class LSTM_GRU_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_GRU_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 68. Transformer + CNN + BiLSTM + GRU
class Transformer_CNN_BiLSTM_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(Transformer_CNN_BiLSTM_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))
        # 通过CNN层提取局部特征
        x = transformer_out.permute(1, 2, 0)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x.permute(1, 0, 2))
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 69. LSTM + Dense + Transformer + GRU
class LSTM_Dense_Transformer_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_Dense_Transformer_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(lstm_out)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(dense_out.permute(1, 0, 2))
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(gru_out[:, -1, :])
        return output

# 70. GRU + Bi-directional GRU + Attention
class GRU_BiGRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU_BiGRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(gru_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bigru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bigru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 71. LSTM + CNN + GRU + Bi-directional GRU
class LSTM_CNN_GRU_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_CNN_GRU_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过CNN层提取局部特征
        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 72. LSTM + GRU + CNN + Attention
class LSTM_GRU_CNN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_GRU_CNN_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(filters, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 73. CNN + Transformer + Bi-directional LSTM + Dense
class CNN_Transformer_BiLSTM_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_Transformer_BiLSTM_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=filters, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim * 2, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x)
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(transformer_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(bilstm_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 74. GRU + CNN + Bi-directional GRU + Attention
class GRU_CNN_BiGRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_BiGRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(x)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bigru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bigru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 75. LSTM + GRU + Transformer + Attention
class LSTM_GRU_Transformer_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(LSTM_GRU_Transformer_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(transformer_out.permute(1, 0, 2)), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * transformer_out.permute(1, 0, 2), dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 76. CNN + GRU + Transformer + Attention
class CNN_GRU_Transformer_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3, num_heads=4, dim_feedforward=128):
        super(CNN_GRU_Transformer_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(gru_out.permute(1, 0, 2))
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(transformer_out.permute(1, 0, 2)), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * transformer_out.permute(1, 0, 2), dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 77. BiLSTM + Transformer + Dense
class BiLSTM_Transformer_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(BiLSTM_Transformer_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=num_heads, dim_feedforward=dim_feedforward)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim * 2, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(bilstm_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(transformer_out.permute(1, 0, 2))
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 78. LSTM + GRU + CNN + Bi-directional GRU
class LSTM_GRU_CNN_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(LSTM_GRU_CNN_BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(x)
        # 取最后一个时间步的输出
        output = self.fc(bigru_out[:, -1, :])
        return output

# 79. GRU + CNN + Attention + Dense
class GRU_CNN_Attention_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(GRU_CNN_Attention_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(hidden_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(filters, 1)
        # Dense层用于特征映射
        self.dense = nn.Linear(filters, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过CNN层提取局部特征
        x = gru_out.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * x, dim=1)
        # 通过Dense层进行特征映射
        dense_out = self.dense(context_vector)
        # 通过全连接层输出最终结果
        output = self.fc(dense_out)
        return output

# 80. LSTM + Bi-directional GRU + Attention
class LSTM_BiGRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_BiGRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Bi-directional GRU层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bigru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过Bi-directional GRU层处理输入序列
        bigru_out, _ = self.bigru(lstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(bigru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * bigru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 81. CNN + LSTM + GRU + Dense
class CNN_LSTM_GRU_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_LSTM_GRU_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(filters, hidden_dim, num_layers, batch_first=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(lstm_out)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 82. BiLSTM + GRU + Attention
class BiLSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 83. Transformer + LSTM + Dense + Attention
class Transformer_LSTM_Dense_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, num_heads=4, dim_feedforward=128):
        super(Transformer_LSTM_Dense_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Transformer Encoder层用于处理全局信息
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Transformer Encoder层处理输入序列
        transformer_out = self.transformer_encoder(x.permute(1, 0, 2))
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(transformer_out.permute(1, 0, 2))
        # 通过Dense层进行特征映射
        dense_out = self.dense(lstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(dense_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * dense_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 84. CNN + GRU + Dense
class CNN_GRU_Dense(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, filters=64, kernel_size=3):
        super(CNN_GRU_Dense, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # CNN层用于提取局部特征
        self.conv = nn.Conv1d(input_dim, filters, kernel_size, padding=(kernel_size-1)//2)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(filters, hidden_dim, num_layers, batch_first=True)
        # Dense层用于特征映射
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过CNN层提取局部特征
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过Dense层进行特征映射
        dense_out = self.dense(gru_out)
        # 取最后一个时间步的输出
        output = self.fc(dense_out[:, -1, :])
        return output

# 85. Bi-directional LSTM + GRU + Attention
class BiLSTM_GRU_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM_GRU_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Bi-directional LSTM层用于处理序列数据,捕捉时间序列中的双向依赖关系
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过Bi-directional LSTM层处理输入序列
        bilstm_out, _ = self.bilstm(x)
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(bilstm_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

# 86. GRU + LSTM + Attention
class GRU_LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU_LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU层用于处理序列数据,捕捉时间序列中的依赖关系
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # LSTM层用于处理序列数据,捕捉时间序列中的依赖关系
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Attention层用于加权序列中的重要信息
        self.attention = nn.Linear(hidden_dim, 1)
        # 全连接层用于最终的分类或回归任务
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过GRU层处理输入序列
        gru_out, _ = self.gru(x)
        # 通过LSTM层处理输入序列
        lstm_out, _ = self.lstm(gru_out)
        # 计算Attention权重
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output
    
    
# 87. CNN + LSTM + Dense  + Attention
class CNN_LSTM_Dense_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers):
        """
        结合Transformer、LSTM、Dense层和多头注意力机制的时间序列预测模型. 
        
        参数:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            nhead (int): 多头注意力的头数
            num_layers (int): Transformer和LSTM的层数
        """
        super(CNN_LSTM_Dense_Attention, self).__init__()
        
        # Transformer层
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        
        # LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        
        # Dense层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数据,形状为 (batch_size, sequence_length, input_dim)
            
        返回:
            out (torch.Tensor): 输出,形状为 (batch_size, sequence_length, output_dim)
        """
        # Transformer编码器
        x_transformer = x.permute(1, 0, 2)
        x_transformer = self.transformer_encoder(x_transformer)
        x_transformer = x_transformer.permute(1, 0, 2)
        
        # LSTM层
        x_lstm, _ = self.lstm(x_transformer)
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(x_lstm, x_lstm, x_lstm)
        
        # Dense层
        out = self.fc(attn_output)
        
        return out
