import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AirQualityVisualizer:
    """
    空气质量数据可视化工具
    功能：读取数据文件，绘制各指标的时间序列图
    """
    def __init__(self):
        # 定义空气质量指标及其对应的颜色
        self.features = {
            'AQI': 'red',
            'PM2.5': 'black',
            'PM10': 'purple',
            'CO': 'brown',
            'SO2': 'blue',
            'NO2': 'green',
            'O3_8h': 'orange'
        }
    
    def load_data(self, file_path):
        """读取数据文件（支持Excel和CSV格式）"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        print(f"数据加载完成，形状: {df.shape}")
        return df
    
    def plot_single_feature(self, df, feature, save_path=None):
        """绘制单个指标的时间序列图"""
        plt.figure(figsize=(10, 6))
        plt.plot(df[feature], color=self.features[feature], linewidth=1.5)
        plt.title(f'{feature} 时间序列')
        plt.xlabel('时间')
        plt.ylabel('浓度')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(f"{save_path}/{feature}_timeseries.png", dpi=300, bbox_inches='tight')
            print(f"已保存 {feature} 时间序列图到 {save_path}/{feature}_timeseries.png")
        
        plt.show()
    
    def plot_all_features(self, df, save_path=None):
        """绘制所有指标的单独时间序列图"""
        for feature in self.features.keys():
            self.plot_single_feature(df, feature, save_path)
    
    def plot_combined_features(self, df, save_path=None):
        """绘制组合时间序列图（第一行4个，第二行3个居中）"""
        # 创建一个2行4列的网格布局
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig)
        
        # 获取特征列表
        features_list = list(self.features.keys())
        
        # 使用科学可视化的颜色方案
        colors = plt.cm.viridis(np.linspace(0, 1, len(features_list)))
        
        # 绘制第一行的4个图
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(df[features_list[i]], color=colors[i], linewidth=1.8)
            ax.set_title(f'{features_list[i]}时间序列', fontsize=12, pad=10)
            ax.set_xlabel('时间', fontsize=10)
            ax.set_ylabel('浓度', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(axis='both', labelsize=9)
            if len(df) > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        
        # 绘制第二行的3个图（从左边开始排列）
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(df[features_list[i+4]], color=colors[i+4], linewidth=1.8)
            ax.set_title(f'{features_list[i+4]}时间序列', fontsize=12, pad=10)
            ax.set_xlabel('时间', fontsize=10)
            ax.set_ylabel('浓度', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(axis='both', labelsize=9)
            if len(df) > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        
        plt.tight_layout(pad=2.0)
        
        if save_path:
            plt.savefig(f"{save_path}/combined_timeseries.png", dpi=300, bbox_inches='tight')
            print(f"已保存组合时间序列图到 {save_path}/combined_timeseries.png")
        
        plt.show()

def main():
    # 创建可视化工具实例
    visualizer = AirQualityVisualizer()
    
    # 设置数据文件路径
    data_file = r"C:\Users\lms\Desktop\2015-2025.xlsx"  # 请替换为实际的数据文件路径
    
    try:
        # 加载数据
        df = visualizer.load_data(data_file)
        
        # 显示数据前几行
        print("\n数据预览:")
        print(df.head())
        
        # 直接绘制组合时间序列图，不保存
        print("\n正在绘制组合时间序列图...")
        visualizer.plot_combined_features(df)
            
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()