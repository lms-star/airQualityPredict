import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# ====================== 基础设置 ======================
# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 强制使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("警告：GPU不可用，将使用CPU进行训练")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型参数设置
SEQUENCE_LENGTH = 10  # 序列长度，即用多少天的数据预测下一天
EPOCHS = 100         # 训练轮次
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 0.001  # 学习率

# ====================== 数据处理类 ======================
class DataProcessor:
    """
    数据预处理类
    功能：加载数据、数据标准化、创建序列、划分数据集
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
    
    def load_data(self, file_path):
        """读取数据文件（支持Excel和CSV格式）"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df['PM2.5'].values  # 只取PM2.5数据
    
    def create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    # 在DataProcessor类中修改process_data方法
    def process_data(self, data):
        """数据预处理主函数"""
        # 只进行一次归一化，将数据缩放到0-1范围
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).ravel()
        
        # 创建序列
        X, y = self.create_sequences(scaled_data)
        X = X.reshape(-1, self.sequence_length, 1)
        y = y.reshape(-1, 1)
        
        # 划分训练集和测试集 (90% 训练, 10% 测试)
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 转换为PyTorch张量
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

# ====================== CNN模型定义 ======================
class PM25CNN(nn.Module):
    """
    PM2.5预测CNN模型
    输入：[batch_size, sequence_length, 1]
    输出：[batch_size, 1]
    """
    def __init__(self, sequence_length):
        super(PM25CNN, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # 计算全连接层的输入维度
        self.flatten_size = 64 * (sequence_length // 4)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # 调整输入维度顺序：[batch, seq_len, 1] -> [batch, 1, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# ====================== 可视化函数 ======================
class Visualizer:
    """可视化工具类"""
    
    def plot_training_loss(self, train_losses, val_losses):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.title('模型训练过程中的损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='真实值', marker='o')
        plt.plot(y_pred, label='预测值', marker='x')
        plt.title('PM2.5预测结果对比')
        plt.xlabel('样本')
        plt.ylabel('PM2.5浓度')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_metrics(self, metrics):
        """绘制评估指标可视化图"""
        plt.figure(figsize=(10, 6))
        names = list(metrics.keys())
        values = list(metrics.values())
        plt.bar(names, values)
        plt.title('模型评估指标')
        plt.xticks(rotation=45)
        plt.ylabel('指标值')
        plt.grid(True)
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
            outputs = model(batch_X)
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
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'轮次 [{epoch+1}/{EPOCHS}], '
                  f'训练损失: {train_loss:.4f}, '
                  f'验证损失: {val_loss:.4f}')
    
    return train_losses, val_losses

# ====================== 评估函数 ======================
def evaluate_model(model, test_loader, scaler):
    """模型评估函数"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # 合并所有批次的预测结果
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)
    
    # 直接在归一化空间计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }
    
    print('\n评估指标（归一化空间）：')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # 只在可视化时进行反归一化
    y_pred_original = scaler.inverse_transform(y_pred)
    y_true_original = scaler.inverse_transform(y_true)
    
    return y_true_original, y_pred_original, metrics

# ====================== 主程序 ======================
def main():
    # 初始化数据处理器
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)
    
    # 加载数据
    file_path = r"C:\Users\lms\Desktop\2015-2025.xlsx"   # 替换为实际的数据文件路径
    data = data_processor.load_data(file_path)
    
    # 数据预处理
    train_loader, test_loader = data_processor.process_data(data)
    
    # 初始化模型
    model = PM25CNN(SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss()
    # 将Adam替换为AdamW，并添加权重衰减参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 训练模型
    print('\n开始训练模型...')
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device
    )
    
    # 评估模型
    print('\n评估模型性能...')
    y_true, y_pred, metrics = evaluate_model(model, test_loader, data_processor.scaler)
    
    # 可视化结果
    visualizer = Visualizer()
    visualizer.plot_training_loss(train_losses, val_losses)
    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_metrics(metrics)
    


# ====================== 模型预测模块 ======================
def predict(model, data_processor, input_data):
    """使用训练好的模型进行预测"""
    model.eval()
    with torch.no_grad():
        # 数据预处理
        scaled_data = data_processor.scaler.transform(input_data.reshape(-1, 1))
        X = torch.FloatTensor(scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)).to(device)
        
        # 预测
        output = model(X)
        prediction = data_processor.scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1))
    
    return prediction[0, 0]

if __name__ == '__main__':
    main()