import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# ====================== 基础设置 ======================
# 设置随机种子，确保结果可复现
torch.manual_seed(50)
np.random.seed(50)

# 强制使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("警告：GPU不可用，将使用CPU进行训练")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型参数设置
SEQUENCE_LENGTH = 10    # 序列长度，即用多少天的数据预测下一天
EPOCHS = 100           # 训练轮次
BATCH_SIZE = 16       # 批次大小
LEARNING_RATE = 0.00005  # 学习率 原值是0.001
HIDDEN_SIZE = 32       # LSTM隐藏层大小
NUM_LAYERS = 2         # LSTM层数

# ====================== 注意力机制模块 ======================
class AttentionLayer(nn.Module):
    """
    注意力机制层
    用于自动学习序列中重要的时间步
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # 加权求和
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights

# ====================== CNN-LSTM-Attention模型定义 ======================
class AirQualityCNNLSTMAtt(nn.Module):
    """
    空气质量预测CNN-LSTM-Attention混合模型
    CNN：提取空间特征
    LSTM：捕捉时间依赖关系
    Attention：关注重要的时间步
    """
    def __init__(self, sequence_length, n_features=7):
        super(AirQualityCNNLSTMAtt, self).__init__()
        
        # CNN部分 - 提取空间特征
        self.cnn = nn.Sequential(
            # 第一个卷积层
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 第二个卷积层
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # LSTM部分 - 处理时序关系
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 注意力层
        self.attention = AttentionLayer(HIDDEN_SIZE * 2)  # 双向LSTM，隐藏层大小翻倍
        
        # 全连接层 - 输出预测
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_features)
        )
    
    def forward(self, x):
        # CNN特征提取
        x = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
        cnn_out = self.cnn(x)
        
        # 准备LSTM输入
        lstm_in = cnn_out.permute(0, 2, 1)  # [batch, channels, seq_len] -> [batch, seq_len, channels]
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_in)
        
        # 注意力机制
        context, attention_weights = self.attention(lstm_out)
        
        # 全连接层预测
        output = self.fc(context)
        return output, attention_weights

# ====================== 数据处理类 ======================
class DataProcessor:
    """
    数据预处理类
    功能：加载数据、数据标准化、创建序列、划分数据集
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scalers = {
            'AQI': MinMaxScaler(),
            'PM2.5': MinMaxScaler(),
            'PM10': MinMaxScaler(),
            'CO': MinMaxScaler(),
            'SO2': MinMaxScaler(),
            'NO2': MinMaxScaler(),
            'O3_8h': MinMaxScaler()
        }
    
    def load_data(self, file_path):
        """读取数据文件（支持Excel和CSV格式）"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        print("数据加载完成，形状:", df.shape)
        return df
    
    def create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def process_data(self, df):
        """数据预处理主函数"""
        features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
        
        # 标准化每个指标的数据
        scaled_data = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            scaled_data[:, i] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            ).ravel()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        print("序列数据形状:", X.shape, y.shape)
        
        # 划分训练集和测试集 (90% 训练, 10% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, random_state=50
        )
        
        # 转换为PyTorch张量并移至GPU
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader

# ====================== 可视化函数 ======================
class Visualizer:
    """可视化工具类"""
    def __init__(self):
        self.features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    
    def plot_training_loss(self, train_losses, val_losses):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', linewidth=2)
        plt.plot(val_losses, label='验证损失', linewidth=2)
        plt.title('模型训练过程中的损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """绘制每个污染物的预测结果对比图"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.ravel()
        
        for idx, (feature, ax) in enumerate(zip(self.features, axes)):
            ax.plot(y_true[:, idx], label='真实值', marker='o')
            ax.plot(y_pred[:, idx], label='预测值', marker='x')
            ax.set_title(f'{feature} 预测结果对比')
            ax.set_xlabel('样本')
            ax.set_ylabel('浓度')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_weights(self, attention_weights, sequence_length):
        """可视化注意力权重"""
        plt.figure(figsize=(12, 8))
        for i, weights in enumerate(attention_weights):
            plt.subplot(len(attention_weights), 1, i + 1)
            plt.bar(range(sequence_length), weights.flatten())
            plt.title(f'样本 {i+1} 的注意力分布')
            plt.xlabel('时间步')
            plt.ylabel('注意力权重')
        plt.tight_layout()
        plt.show()

# ====================== 训练函数 ======================
def train_model(model, train_loader, test_loader, criterion, optimizer, device):
    """模型训练函数"""
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# ====================== 评估函数 ======================
def evaluate_model(model, test_loader, device):
    """模型评估函数"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs, attention_weights = model(batch_X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # 合并所有批次的预测结果
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    attention_weights = np.vstack(all_attention_weights)
    
    # 计算每个指标的评估指标
    features = ['AQI', 'PM2.5', 'PM10', 'CO', 'SO2', 'NO2', 'O3_8h']
    for i, feature in enumerate(features):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        # 【新增】计算RMSE指标：利用均方误差求平方根
        rmse = np.sqrt(mse)  # 修改位置
        print(f'{feature}:')
        print(f'  RMSE: {rmse:.4f}')  # 修改位置
        print(f'  MSE: {mse:.4f}')
        print(f'  MAE: {mae:.4f}')
        print(f'  R2 Score: {r2:.4f}')
    
    return y_true, y_pred, attention_weights


# ====================== 主函数补全 ======================
def main(data_file):
    """主函数"""
    # 初始化数据处理器
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)

    # 加载并处理数据
    df = data_processor.load_data(data_file)
    train_loader, test_loader = data_processor.process_data(df)  # 补全数据预处理流程

    # 模型初始化
    model = AirQualityCNNLSTMAtt(
        sequence_length=SEQUENCE_LENGTH,
        n_features=7
    ).to(device)

    # 打印模型结构
    print("模型结构：")
    print(model)

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.HuberLoss()  # 使用鲁棒性更好的损失函数

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, device
    )

    # 可视化训练过程
    visualizer = Visualizer()
    visualizer.plot_training_loss(train_losses, val_losses)

    # 模型评估
    print("\n开始评估...")
    y_true, y_pred, attention_weights = evaluate_model(model, test_loader, device)

    # 反标准化数据
    def inverse_transform(data, feature_idx):
        return data_processor.scalers[visualizer.features[feature_idx]].inverse_transform(data.reshape(-1, 1))

    # 反标准化所有特征
    y_true_orig = np.hstack([inverse_transform(y_true[:, i], i) for i in range(7)])
    y_pred_orig = np.hstack([inverse_transform(y_pred[:, i], i) for i in range(7)])

    # 可视化预测结果
    visualizer.plot_predictions(y_true_orig, y_pred_orig)

    # 可视化注意力权重（随机选择5个样本）
    sample_weights = attention_weights[:5]
    visualizer.plot_attention_weights(sample_weights, SEQUENCE_LENGTH)

    # 保存模型
    torch.save(model.state_dict(), "best_model.pth")
    print("\n模型已保存为best_model.pth")


# ====================== 执行入口 ======================
if __name__ == "__main__":
    data_path = r"C:\Users\lms\Desktop\2015-2025.xlsx"  # 替换为实际数据文件路径
    try:
        main(data_path)
    except Exception as e:
        print(f"运行错误：{str(e)}")
        sys.exit(1)