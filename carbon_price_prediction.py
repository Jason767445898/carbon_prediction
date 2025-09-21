#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碳价格预测综合分析系统
整合LSTM、Transformer Attention和SHAP可解释性分析

主要功能：
1. 从Excel文件加载碳价格和相关因子数据
2. 使用LSTM和Transformer模型进行时间序列预测
3. 通过SHAP分析模型决策的可解释性
4. 输出预测结果、准确度评估和解释性分析报告

使用说明：详见《碳价格预测系统使用指南.md》
"""

# =============================================================================
# 全局配置和路径设置
# =============================================================================

# 默认数据文件路径
DEFAULT_DATA_FILE = 'carbon_price_prediction_test_data.xlsx'
SAMPLE_DATA_FILE = 'carbon_price_prediction_test_data.xlsx'

# 输出目录配置
OUTPUT_DIRS = {
    'txt': 'carbon_prediction_log_txt',
    'excel': 'carbon_prediction_results_excel', 
    'pic': 'carbon_prediction_results_pic'
}

# 文件名格式配置
FILE_NAME_FORMAT = {
    'program_name': 'carbon_price_prediction',
    'timestamp_format': '%Y%m%d_%H%M%S'
}

# 默认系统配置
DEFAULT_CONFIG = {
    'target_column': 'carbon_price',
    'sequence_length': 60,
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'lstm_config': {
        'units': [64, 32],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32
    },
    'transformer_config': {
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'dff': 512,
        'dropout': 0.1,
        'epochs': 50
    }
}

# =============================================================================
# 导入必要的库
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

# 机器学习和深度学习
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 深度学习库
try:
    import tensorflow as tf  # type: ignore
    # 统一使用tf.keras
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
    keras = tf.keras  # 为了兼容性
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow不可用: {e}，将跳过深度学习模型")
    # 创建虚拟对象避免类型检查错误
    class DummyTF:
        @staticmethod
        def random():
            return type('obj', (object,), {'set_seed': lambda x: None})()
        
        @staticmethod
        def keras():
            return type('obj', (object,), {
                'Model': object,
                'optimizers': type('obj', (object,), {'Adam': object})()
            })()
    
    tf = DummyTF()  # type: ignore
    keras = DummyTF().keras()  # type: ignore

# 可解释性分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP不可用，将跳过可解释性分析")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost不可用，将使用基础模型")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

print("碳价格预测系统初始化完成")
if TENSORFLOW_AVAILABLE:
    print(f"TensorFlow版本: {tf.__version__}")
if SHAP_AVAILABLE:
    print(f"SHAP版本: {shap.__version__}")


class CarbonPricePredictionSystem:
    """碳价格预测系统主类"""
    
    def __init__(self, config=None, output_dir=None):
        """初始化系统配置"""
        self.config = config or DEFAULT_CONFIG.copy()
        self.data = None
        self.processed_data = None
        self.models = {}
        self.predictions = {}
        self.shap_values = {}
        self.feature_names = []
        self.scalers = {}
        
        # 设置输出目录结构
        self.base_dir = output_dir or "."
        self.program_name = FILE_NAME_FORMAT['program_name']
        self.run_timestamp = datetime.now().strftime(FILE_NAME_FORMAT['timestamp_format'])
        self.run_name = f"{self.program_name}_{self.run_timestamp}"
        
        # 创建输出目录
        self.output_dirs = {
            key: os.path.join(self.base_dir, path)
            for key, path in OUTPUT_DIRS.items()
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
    def _default_config(self):
        """
        默认配置
        
        🔧 配置参数说明：
        =================
        
        target_column: 目标列名（碳价格列）
        - 默认: 'carbon_price'
        - 如果你的数据列名不同，需要修改此参数
        - 常见列名: 'price', '碳价格', 'carbon_price_eur', 'emission_price' 等
        
        sequence_length: 时间序列长度
        - 默认: 60 (天)
        - 用于LSTM和Transformer模型的输入序列长度
        - 建议范围: 30-120，取决于数据频率和预测时间跨度
        
        test_size: 测试集比例
        - 默认: 0.2 (20%)
        - 用于最终模型性能评估的数据比例
        
        validation_size: 验证集比例
        - 默认: 0.1 (10%)
        - 用于模型训练过程中的验证和调参
        
        lstm_config: LSTM模型配置
        - units: 隐藏层单元数 [64, 32]
        - dropout: 随机失活率 0.2
        - epochs: 训练轮数 100
        - batch_size: 批次大小 32
        
        transformer_config: Transformer模型配置
        - d_model: 模型维度 128
        - num_heads: 注意力头数 8
        - num_layers: 编码器层数 4
        - dff: 前馈网络维度 512
        - dropout: 随机失活率 0.1
        - epochs: 训练轮数 50
        
        💡 如何自定义配置：
        --------------------
        custom_config = {
            'target_column': '你的列名',     # 修改目标列名
            'sequence_length': 90,           # 增加序列长度
            'test_size': 0.15,              # 调整测试集比例
            'lstm_config': {
                'units': [128, 64, 32],      # 更复杂的网络结构
                'epochs': 200,               # 更多训练轮数
                'batch_size': 16             # 更小的批次大小
            }
        }
        system = CarbonPricePredictionSystem(config=custom_config)
        """
        return {
            'target_column': 'carbon_price',
            'sequence_length': 60,
            'test_size': 0.2,
            'validation_size': 0.1,
            'lstm_config': {
                'units': [64, 32],
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32
            },
            'transformer_config': {
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 4,
                'dff': 512,
                'dropout': 0.1,
                'epochs': 50
            }
        }
    
    def load_data(self, file_path, sheet_name=None):
        try:
            print(f"正在加载数据文件: {file_path}")
            
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                # 读取Excel文件，处理多工作表情况
                excel_data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, parse_dates=True)
                
                # 如果sheet_name=None，会返回字典，需要选择第一个工作表
                if isinstance(excel_data, dict):
                    # 获取第一个工作表
                    first_sheet_name = list(excel_data.keys())[0]
                    self.data = excel_data[first_sheet_name]
                    print(f"检测到多个工作表，自动选择第一个工作表: {first_sheet_name}")
                else:
                    self.data = excel_data
            elif file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else:
                raise ValueError("支持的文件格式: .xlsx, .xls, .csv")
            
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"数据列: {list(self.data.columns)}")
            print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            
            # 检查必要列
            if self.config['target_column'] not in self.data.columns:
                available_cols = list(self.data.columns)
                print(f"警告: 未找到目标列 '{self.config['target_column']}'")
                print(f"可用列: {available_cols}")
                # 尝试猜测碳价格列
                possible_cols = [col for col in available_cols if any(keyword in col.lower() for keyword in ['price', 'carbon', '价格', '碳'])]
                if possible_cols:
                    self.config['target_column'] = possible_cols[0]
                    print(f"自动选择目标列: {self.config['target_column']}")
                else:
                    raise ValueError(f"请指定正确的碳价格列名，可用列: {available_cols}")
            
            return self.data
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
    
    def create_sample_data(self, start_date='2020-01-01', end_date='2023-12-31', save_path=None):
        """
        创建示例碳价格数据
        
        📊 此函数用于生成演示数据，如果你有自己的数据，可以跳过此步
        ================================================================
        
        参数:
            start_date: 开始日期（默认:'2020-01-01'）
            end_date: 结束日期（默认:'2023-12-31'）
            save_path: 保存路径（可选）
        
        🔧 生成的示例数据包含：
        ---------------------------
        - carbon_price: 碳价格（目标变量）
        - gdp_growth: GDP增长率
        - industrial_production: 工业生产指数
        - oil_price: 石油价格
        - gas_price: 天然气价格
        - electricity_demand: 电力需求
        - temperature: 温度
        - policy_impact: 政策影响指数
        - tech_innovation: 技术创新指数
        - emissions: 碳排放量
        - 以及各种技术指标（滞后、移动平均、波动率等）
        
        💡 如果你有自己的数据：
        -------------------------
        1. 跳过此函数，直接使用 load_data() 加载你的数据
        2. 确保你的数据包含类似的列结构
        3. 或者参考此函数生成的数据格式来准备你的数据
        """
        print("创建示例碳价格数据...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 模拟碳价格数据（参考欧盟ETS等碳市场）
        base_price = 50  # 基础价格
        
        # 趋势分量（长期上升趋势）
        trend = np.linspace(0, 30, n_days)
        
        # 季节性分量
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + \
                  2 * np.sin(2 * np.pi * np.arange(n_days) / (365.25/12))
        
        # 随机波动
        noise = np.random.normal(0, 3, n_days)
        
        # 价格冲击事件
        shock_days = np.random.choice(n_days, size=10, replace=False)
        shocks = np.zeros(n_days)
        for day in shock_days:
            shocks[day:day+5] = np.random.normal(0, 8)
        
        carbon_price = base_price + trend + seasonal + noise + shocks
        carbon_price = np.maximum(carbon_price, 5)  # 确保价格不为负
        
        # 相关影响因子
        data = pd.DataFrame(index=dates)
        data['carbon_price'] = carbon_price
        
        # GDP增长率影响
        gdp_growth = 2 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 1, n_days)
        data['gdp_growth'] = gdp_growth
        
        # 工业生产指数
        industrial_production = 100 + np.cumsum(np.random.normal(0.01, 0.5, n_days))
        data['industrial_production'] = industrial_production
        
        # 能源价格（石油、天然气）
        oil_price = 60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 180) + np.random.normal(0, 5, n_days)
        data['oil_price'] = np.maximum(oil_price, 20)
        
        gas_price = 3 + 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 120) + np.random.normal(0, 0.8, n_days)
        data['gas_price'] = np.maximum(gas_price, 1)
        
        # 电力需求
        electricity_demand = 1000 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + \
                           50 * np.sin(2 * np.pi * np.arange(n_days) / 7) + np.random.normal(0, 30, n_days)
        data['electricity_demand'] = electricity_demand
        
        # 温度（影响能源需求）
        temperature = 15 + 10 * np.sin(2 * np.pi * (np.arange(n_days) - 80) / 365.25) + np.random.normal(0, 3, n_days)
        data['temperature'] = temperature
        
        # 政策指数（模拟政策影响）
        policy_impact = np.cumsum(np.random.choice([0, 0, 0, 1, -1], n_days, p=[0.7, 0.1, 0.1, 0.05, 0.05]))
        data['policy_impact'] = policy_impact
        
        # 技术创新指数
        tech_innovation = np.cumsum(np.random.exponential(0.1, n_days))
        data['tech_innovation'] = tech_innovation
        
        # 碳排放量
        emissions = 1000 - 0.1 * tech_innovation + np.random.normal(0, 50, n_days)
        data['emissions'] = np.maximum(emissions, 500)
        
        # 添加滞后变量和技术指标
        data['carbon_price_lag1'] = data['carbon_price'].shift(1)
        data['carbon_price_lag7'] = data['carbon_price'].shift(7)
        data['carbon_price_ma7'] = data['carbon_price'].rolling(7).mean()
        data['carbon_price_ma30'] = data['carbon_price'].rolling(30).mean()
        data['price_volatility'] = data['carbon_price'].rolling(14).std()
        
        # 移除NaN值
        data = data.dropna()
        
        self.data = data
        print(f"示例数据创建完成，形状: {data.shape}")
        
        if save_path:
            save_file = os.path.join(self.output_dirs['excel'], f"{self.run_name}_sample_data.xlsx")
            data.to_excel(save_file)
            print(f"数据已保存到: {save_file}")
        
        return data
    
    def preprocess_data(self):
        """数据预处理和特征工程"""
        print("开始数据预处理...")
        
        if self.data is None:
            raise ValueError("请先加载数据")
        
        df = self.data.copy()
        
        # 基础特征工程
        target_col = self.config['target_column']
        
        # 价格变化特征
        df['price_return'] = df[target_col].pct_change()
        df['price_diff'] = df[target_col].diff()
        
        # 移动平均特征
        for window in [5, 10, 20, 30]:
            df[f'ma_{window}'] = df[target_col].rolling(window).mean()
            df[f'ma_{window}_ratio'] = df[target_col] / df[f'ma_{window}']
        
        # 波动率特征
        for window in [7, 14, 30]:
            df[f'volatility_{window}'] = df['price_return'].rolling(window).std()
        
        # 技术指标
        # RSI
        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        bb_window = 20
        df['bb_middle'] = df[target_col].rolling(bb_window).mean()
        bb_std = df[target_col].rolling(bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df[target_col] - df['bb_lower']) / df['bb_width']
        
        # 价格动量
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df[target_col].diff(period)
        
        # 滞后特征
        for lag in [1, 2, 3, 5, 10]:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # 移除无效值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # 选择特征列
        exclude_cols = [target_col, 'price_return', 'price_diff']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        self.processed_data = df
        
        print(f"预处理完成，数据形状: {df.shape}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"特征列表: {feature_cols}")
        
        return df
    
    def prepare_sequences(self, data, target_col, feature_cols, seq_length):
        """准备序列数据用于LSTM和Transformer"""
        sequences = []
        targets = []
        
        for i in range(seq_length, len(data)):
            seq = data[feature_cols].iloc[i-seq_length:i].values
            target = data[target_col].iloc[i]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def split_data(self):
        """分割训练、验证和测试数据"""
        if self.processed_data is None:
            raise ValueError("请先进行数据预处理")
        
        target_col = self.config['target_column']
        test_size = self.config['test_size']
        val_size = self.config['validation_size']
        
        # 时间序列分割（保持时间顺序）
        n = len(self.processed_data)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_data = self.processed_data.iloc[:train_end]
        val_data = self.processed_data.iloc[train_end:val_end]
        test_data = self.processed_data.iloc[val_end:]
        
        print(f"数据分割完成:")
        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")
        
        return train_data, val_data, test_data
    
    def build_lstm_model(self):
        """构建LSTM模型"""
        print("构建LSTM模型...")
        
        config = self.config['lstm_config']
        seq_length = self.config['sequence_length']
        n_features = len(self.feature_names)
        
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(
            units=config['units'][0],
            return_sequences=True,
            input_shape=(seq_length, n_features)
        ))
        model.add(Dropout(config['dropout']))
        
        # 第二层LSTM
        if len(config['units']) > 1:
            model.add(LSTM(
                units=config['units'][1],
                return_sequences=False
            ))
            model.add(Dropout(config['dropout']))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"LSTM模型架构:")
        model.summary()
        
        return model
    
    def build_transformer_model(self):
        """构建Transformer模型"""
        print("构建Transformer模型...")
        
        config = self.config['transformer_config']
        seq_length = self.config['sequence_length']
        n_features = len(self.feature_names)
        
        # 输入层
        inputs = layers.Input(shape=(seq_length, n_features))
        
        # 投影到d_model维度
        x = layers.Dense(config['d_model'])(inputs)
        
        # 位置编码
        x = self._add_positional_encoding(x, seq_length, config['d_model'])
        
        # Transformer编码器层
        for _ in range(config['num_layers']):
            x = self._transformer_encoder(
                x, 
                config['d_model'], 
                config['num_heads'], 
                config['dff'],
                config['dropout']
            )
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 输出层
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # 自定义学习率调度
        learning_rate = self._create_lr_schedule(config['d_model'])
        optimizer = keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Transformer模型架构:")
        model.summary()
        
        return model
    
    def _add_positional_encoding(self, x, seq_len, d_model):
        """添加位置编码"""
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates
        
        angle_rads = get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        
        return x + pos_encoding
    
    def _transformer_encoder(self, x, d_model, num_heads, dff, dropout_rate):
        """Transformer编码器层"""
        # 多头注意力
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # 前馈网络
        ffn_output = layers.Dense(dff, activation='relu')(out1)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def _create_lr_schedule(self, d_model, warmup_steps=4000):
        """创建学习率调度"""
        class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)
                self.warmup_steps = warmup_steps
            
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
        return CustomSchedule(d_model, warmup_steps)
    
    def build_ml_models(self):
        """构建机器学习模型用于SHAP分析"""
        print("构建机器学习模型...")
        
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        }
        
        # 只在XGBoost可用时添加
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        return models
    
    def train_models(self):
        """训练所有模型"""
        print("="*60)
        print("开始训练模型")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("请先进行数据预处理")
        
        # 分割数据
        train_data, val_data, test_data = self.split_data()
        
        target_col = self.config['target_column']
        seq_length = self.config['sequence_length']
        
        # 准备序列数据
        X_seq_train, y_seq_train = self.prepare_sequences(
            train_data, target_col, self.feature_names, seq_length
        )
        X_seq_val, y_seq_val = self.prepare_sequences(
            val_data, target_col, self.feature_names, seq_length
        )
        X_seq_test, y_seq_test = self.prepare_sequences(
            test_data, target_col, self.feature_names, seq_length
        )
        
        # 准备机器学习数据
        X_ml_train = train_data[self.feature_names].values
        y_ml_train = train_data[target_col].values
        X_ml_test = test_data[self.feature_names].values
        y_ml_test = test_data[target_col].values
        
        # 数据标准化
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        
        X_seq_train_scaled = scaler_features.fit_transform(
            X_seq_train.reshape(-1, X_seq_train.shape[-1])
        ).reshape(X_seq_train.shape)
        X_seq_val_scaled = scaler_features.transform(
            X_seq_val.reshape(-1, X_seq_val.shape[-1])
        ).reshape(X_seq_val.shape)
        X_seq_test_scaled = scaler_features.transform(
            X_seq_test.reshape(-1, X_seq_test.shape[-1])
        ).reshape(X_seq_test.shape)
        
        y_seq_train_scaled = scaler_target.fit_transform(y_seq_train.reshape(-1, 1)).flatten()
        y_seq_val_scaled = scaler_target.transform(y_seq_val.reshape(-1, 1)).flatten()
        
        # 保存缩放器
        self.scalers = {
            'features': scaler_features,
            'target': scaler_target
        }
        
        # 训练LSTM模型
        print("\n训练LSTM模型...")
        lstm_model = self.build_lstm_model()
        
        lstm_history = lstm_model.fit(
            X_seq_train_scaled, y_seq_train_scaled,
            validation_data=(X_seq_val_scaled, y_seq_val_scaled),
            epochs=self.config['lstm_config']['epochs'],
            batch_size=self.config['lstm_config']['batch_size'],
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=5, factor=0.5
                )
            ]
        )
        
        # 训练Transformer模型
        print("\n训练Transformer模型...")
        transformer_model = self.build_transformer_model()
        
        transformer_history = transformer_model.fit(
            X_seq_train_scaled, y_seq_train_scaled,
            validation_data=(X_seq_val_scaled, y_seq_val_scaled),
            epochs=self.config['transformer_config']['epochs'],
            batch_size=32,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                )
            ]
        )
        
        # 训练机器学习模型
        print("\n训练机器学习模型...")
        ml_models = self.build_ml_models()
        
        # 对ML模型的特征进行标准化
        scaler_ml = StandardScaler()
        X_ml_train_scaled = scaler_ml.fit_transform(X_ml_train)
        X_ml_test_scaled = scaler_ml.transform(X_ml_test)
        self.scalers['ml_features'] = scaler_ml
        
        for name, model in ml_models.items():
            print(f"训练 {name}...")
            model.fit(X_ml_train_scaled, y_ml_train)
        
        # 保存模型和数据
        self.models = {
            'lstm': lstm_model,
            'transformer': transformer_model,
            **ml_models
        }
        
        self.train_data = {
            'X_seq_train': X_seq_train_scaled,
            'y_seq_train': y_seq_train_scaled,
            'X_seq_test': X_seq_test_scaled,
            'y_seq_test': y_seq_test,
            'X_ml_train': X_ml_train_scaled,
            'y_ml_train': y_ml_train,
            'X_ml_test': X_ml_test_scaled,
            'y_ml_test': y_ml_test
        }
        
        self.train_history = {
            'lstm': lstm_history,
            'transformer': transformer_history
        }
        
        print("\n所有模型训练完成！")
        
        return self.models
    
    def evaluate_models(self):
        """评估所有模型性能"""
        print("="*60)
        print("模型性能评估")
        print("="*60)
        
        if not self.models:
            raise ValueError("请先训练模型")
        
        results = {}
        
        # 深度学习模型预测
        for model_name in ['lstm', 'transformer']:
            if model_name in self.models:
                print(f"\n评估 {model_name.upper()} 模型...")
                model = self.models[model_name]
                
                # 预测
                y_pred_scaled = model.predict(
                    self.train_data['X_seq_test'], verbose=0
                )
                
                # 反标准化
                y_pred = self.scalers['target'].inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
                y_true = self.train_data['y_seq_test']
                
                # 计算指标
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # 方向准确率
                direction_accuracy = np.mean(
                    np.sign(y_pred[1:] - y_pred[:-1]) == 
                    np.sign(y_true[1:] - y_true[:-1])
                ) * 100
                
                results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'Direction_Accuracy': direction_accuracy,
                    'predictions': y_pred,
                    'actual': y_true
                }
                
                print(f"MSE: {mse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R²: {r2:.4f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"方向准确率: {direction_accuracy:.2f}%")
        
        # 机器学习模型预测
        for model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            if model_name in self.models:
                print(f"\n评估 {model_name} 模型...")
                model = self.models[model_name]
                
                # 预测
                y_pred = model.predict(self.train_data['X_ml_test'])
                y_true = self.train_data['y_ml_test']
                
                # 计算指标
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                # 方向准确率
                direction_accuracy = np.mean(
                    np.sign(y_pred[1:] - y_pred[:-1]) == 
                    np.sign(y_true[1:] - y_true[:-1])
                ) * 100
                
                results[model_name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'Direction_Accuracy': direction_accuracy,
                    'predictions': y_pred,
                    'actual': y_true
                }
                
                print(f"MSE: {mse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R²: {r2:.4f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"方向准确率: {direction_accuracy:.2f}%")
        
        self.predictions = results
        
        # 创建性能对比表
        performance_df = pd.DataFrame({
            model: {
                'MSE': result['MSE'],
                'MAE': result['MAE'],
                'RMSE': result['RMSE'],
                'R²': result['R²'],
                'MAPE(%)': result['MAPE'],
                '方向准确率(%)': result['Direction_Accuracy']
            }
            for model, result in results.items()
        }).T
        
        print("\n\n模型性能对比:")
        print(performance_df.round(4))
        
        # 找出最佳模型
        best_model = performance_df['R²'].idxmax()
        print(f"\n最佳模型（基于R²）: {best_model}")
        
        return results, performance_df
    
    def perform_shap_analysis(self):
        """执行SHAP可解释性分析"""
        print("="*60)
        print("SHAP可解释性分析")
        print("="*60)
        
        if not self.models:
            raise ValueError("请先训练模型")
        
        # 选择最佳的机器学习模型进行SHAP分析
        ml_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
        available_ml_models = [name for name in ml_models if name in self.models]
        
        if not available_ml_models:
            print("没有可用的机器学习模型进行SHAP分析")
            return
        
        # 选择最佳模型（基于R²）
        if not hasattr(self, 'predictions'):
            print("请先运行模型评估")
            return
        
        ml_results = {name: result for name, result in self.predictions.items() 
                     if name in available_ml_models}
        
        if not ml_results:
            print("没有机器学习模型的预测结果")
            return
        
        best_ml_model = max(ml_results.keys(), 
                           key=lambda x: ml_results[x]['R²'])
        model = self.models[best_ml_model]
        
        print(f"选择 {best_ml_model} 模型进行SHAP分析...")
        
        # 准备SHAP分析数据
        X_train = self.train_data['X_ml_train']
        X_test = self.train_data['X_ml_test']
        
        # 创建SHAP解释器
        print("创建SHAP解释器...")
        if best_ml_model in ['RandomForest', 'XGBoost']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train[:100])  # 使用样本作为背景
        
        # 计算SHAP值
        print("计算SHAP值...")
        shap_values = explainer.shap_values(X_test)
        
        # 如果是多输出，取第一个输出的SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print("\n特征重要性排序:")
        print(feature_importance.head(10))
        
        # 保存SHAP分析结果
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'model_name': best_ml_model
        }
        
        return self.shap_values
    
    def create_visualizations(self, save_dir=None):
        """创建可视化图表"""
        print("="*60)
        print("创建可视化图表")
        print("="*60)
        
        if save_dir is None:
            save_dir = self.output_dirs['pic']
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 模型性能对比
        if hasattr(self, 'predictions'):
            self._plot_model_performance(save_dir)
        
        # 2. 预测结果对比
        if hasattr(self, 'predictions'):
            self._plot_predictions(save_dir)
        
        # 3. SHAP分析图表
        if hasattr(self, 'shap_values'):
            self._plot_shap_analysis(save_dir)
        
        # 4. 训练历史
        if hasattr(self, 'train_history'):
            self._plot_training_history(save_dir)
        
        print(f"所有图表已保存到: {save_dir}")
    
    def _plot_model_performance(self, save_dir):
        """绘制模型性能对比图"""
        if not hasattr(self, 'predictions'):
            return
        
        # 准备数据
        models = list(self.predictions.keys())
        metrics = ['MSE', 'MAE', 'R²', 'MAPE', 'Direction_Accuracy']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [self.predictions[model][metric] for model in models]
                
                ax = axes[i]
                bars = ax.bar(models, values, alpha=0.7)
                ax.set_title(f'{metric} 对比', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
        
        # 移除多余的子图
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_model_performance_comparison.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_predictions(self, save_dir):
        """绘制预测结果对比图"""
        if not hasattr(self, 'predictions'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 选择最好的几个模型进行展示
        models_to_show = list(self.predictions.keys())[:4]
        
        for i, model_name in enumerate(models_to_show):
            ax = axes[i//2, i%2]
            
            actual = self.predictions[model_name]['actual']
            predicted = self.predictions[model_name]['predictions']
            
            # 只显示最后200个点以提高可读性
            if len(actual) > 200:
                actual = actual[-200:]
                predicted = predicted[-200:]
            
            ax.plot(actual, label='实际值', linewidth=2, alpha=0.8)
            ax.plot(predicted, label='预测值', linewidth=2, alpha=0.8)
            
            ax.set_title(f'{model_name} 预测结果', fontsize=14, fontweight='bold')
            ax.set_xlabel('时间步')
            ax.set_ylabel('碳价格')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加R²信息
            r2 = self.predictions[model_name]['R²']
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_prediction_comparison.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_shap_analysis(self, save_dir):
        """绘制SHAP分析图表"""
        if not hasattr(self, 'shap_values'):
            return
        
        shap_vals = self.shap_values['values']
        X_test = self.train_data['X_ml_test']
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_test, feature_names=self.feature_names, 
                         show=False)
        plt.title('SHAP特征重要性总结图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_shap_summary_plot.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. SHAP Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, feature_names=self.feature_names,
                         plot_type="bar", show=False)
        plt.title('SHAP特征重要性条形图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_shap_bar_plot.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 前几个重要特征的依赖图
        top_features = self.shap_values['feature_importance'].head(4)['Feature'].tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                feature_idx = self.feature_names.index(feature)
                
                plt.sca(axes[i])
                shap.dependence_plot(feature_idx, shap_vals, X_test,
                                   feature_names=self.feature_names, 
                                   show=False)
                axes[i].set_title(f'{feature} 依赖图', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_shap_dependence_plots.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_training_history(self, save_dir):
        """绘制训练历史"""
        if not hasattr(self, 'train_history'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # LSTM训练历史
        if 'lstm' in self.train_history:
            history = self.train_history['lstm']
            
            axes[0, 0].plot(history.history['loss'], label='训练损失')
            axes[0, 0].plot(history.history['val_loss'], label='验证损失')
            axes[0, 0].set_title('LSTM损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(history.history['mae'], label='训练MAE')
            axes[0, 1].plot(history.history['val_mae'], label='验证MAE')
            axes[0, 1].set_title('LSTM MAE曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Transformer训练历史
        if 'transformer' in self.train_history:
            history = self.train_history['transformer']
            
            axes[1, 0].plot(history.history['loss'], label='训练损失')
            axes[1, 0].plot(history.history['val_loss'], label='验证损失')
            axes[1, 0].set_title('Transformer损失曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(history.history['mae'], label='训练MAE')
            axes[1, 1].plot(history.history['val_mae'], label='验证MAE')
            axes[1, 1].set_title('Transformer MAE曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pic_file = os.path.join(save_dir, f'{self.run_name}_training_history.png')
        plt.savefig(pic_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path=None):
        """生成详细的分析报告"""
        print("="*60)
        print("生成分析报告")
        print("="*60)
        
        # 生成详细的文本报告
        self._generate_detailed_text_report()
        
        # 生成运行日志
        self._generate_runtime_log()
        
        if save_path is None:
            save_path = os.path.join(self.output_dirs['excel'], f'{self.run_name}_report.xlsx')
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            
            # 1. 数据概要
            if self.processed_data is not None:
                data_summary = self.processed_data.describe()
                data_summary.to_excel(writer, sheet_name='数据概要')
            
            # 2. 模型性能对比
            if hasattr(self, 'predictions'):
                performance_df = pd.DataFrame({
                    model: {
                        'MSE': result['MSE'],
                        'MAE': result['MAE'],
                        'RMSE': result['RMSE'],
                        'R²': result['R²'],
                        'MAPE(%)': result['MAPE'],
                        '方向准确率(%)': result['Direction_Accuracy']
                    }
                    for model, result in self.predictions.items()
                }).T
                performance_df.to_excel(writer, sheet_name='模型性能')
            
            # 3. 特征重要性
            if hasattr(self, 'shap_values'):
                feature_importance = self.shap_values['feature_importance']
                feature_importance.to_excel(writer, sheet_name='特征重要性', index=False)
            
            # 4. 预测结果（选择最佳模型）
            if hasattr(self, 'predictions'):
                best_model = max(self.predictions.keys(), 
                               key=lambda x: self.predictions[x]['R²'])
                
                predictions_df = pd.DataFrame({
                    '实际值': self.predictions[best_model]['actual'],
                    '预测值': self.predictions[best_model]['predictions'],
                    '误差': (self.predictions[best_model]['actual'] - 
                             self.predictions[best_model]['predictions'])
                })
                predictions_df.to_excel(writer, sheet_name=f'{best_model}_预测结果', index=False)
            
            # 5. 系统配置信息
            config_df = pd.DataFrame([
                ['目标列', self.config.get('target_column', 'carbon_price')],
                ['序列长度', self.config.get('sequence_length', 60)],
                ['测试集比例', self.config.get('test_size', 0.2)],
                ['随机种子', self.config.get('random_state', 42)],
                ['数据点数量', len(self.processed_data) if self.processed_data is not None else 0],
                ['特征数量', len(self.feature_names) if hasattr(self, 'feature_names') else 0],
                ['生成时间', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            ], columns=['配置项', '数值'])
            config_df.to_excel(writer, sheet_name='系统配置', index=False)
            
            # 6. 数据质量分析
            if self.processed_data is not None:
                quality_df = pd.DataFrame([
                    ['数据完整性', f"{(1 - self.processed_data.isnull().sum().sum() / (len(self.processed_data) * len(self.processed_data.columns))) * 100:.1f}%"],
                    ['时间范围', f"{self.processed_data.index[0].strftime('%Y-%m-%d')} 到 {self.processed_data.index[-1].strftime('%Y-%m-%d')}"],
                    ['价格波动率', f"{self.processed_data[self.config.get('target_column', 'carbon_price')].std():.4f}"],
                    ['价格范围', f"{self.processed_data[self.config.get('target_column', 'carbon_price')].min():.2f} - {self.processed_data[self.config.get('target_column', 'carbon_price')].max():.2f}"],
                    ['数据来源', self.data_source if hasattr(self, 'data_source') else '示例数据']
                ], columns=['质量指标', '数值'])
                quality_df.to_excel(writer, sheet_name='数据质量', index=False)
        
        print(f"Excel报告已保存到: {save_path}")
        print(f"详细文本报告已保存到: {os.path.join(self.output_dirs['txt'], f'{self.run_name}_detailed_report.txt')}")
        print(f"运行日志已保存到: {os.path.join(self.output_dirs['txt'], f'{self.run_name}_runtime_log.txt')}")
        
        # 打印简要报告
        self._print_summary_report()
    
    def _generate_detailed_text_report(self):
        """生成详细的文本报告"""
        report_content = []
        report_content.append("=" * 80)
        report_content.append("碳价格预测系统详细分析报告")
        report_content.append("=" * 80)
        report_content.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 数据概要
        report_content.append("📈 数据信息:")
        if self.processed_data is not None:
            report_content.append(f"   • 数据点数量: {len(self.processed_data):,}")
            report_content.append(f"   • 特征数量: {len(self.feature_names)}")
            report_content.append(f"   • 时间范围: {self.processed_data.index[0].strftime('%Y-%m-%d')} 到 {self.processed_data.index[-1].strftime('%Y-%m-%d')}")
            target_col = self.config.get('target_column', 'carbon_price')
            if target_col in self.processed_data.columns:
                report_content.append(f"   • 碳价格范围: {self.processed_data[target_col].min():.2f} - {self.processed_data[target_col].max():.2f}")
        report_content.append("")
        
        # 模型性能
        if hasattr(self, 'predictions'):
            report_content.append("🏆 模型性能详情:")
            sorted_models = sorted(self.predictions.items(), 
                                 key=lambda x: x[1]['R²'], reverse=True)
            
            for i, (model, result) in enumerate(sorted_models, 1):
                report_content.append(f"   {i}. {model}:")
                report_content.append(f"      • R²: {result['R²']:.4f}")
                report_content.append(f"      • RMSE: {result['RMSE']:.4f}")
                report_content.append(f"      • MAE: {result['MAE']:.4f}")
                report_content.append(f"      • MAPE: {result['MAPE']:.2f}%")
                report_content.append(f"      • 方向准确率: {result['Direction_Accuracy']:.2f}%")
                
                # 性能评估
                if result['R²'] > 0.8:
                    performance_level = "优秀"
                elif result['R²'] > 0.6:
                    performance_level = "良好"
                else:
                    performance_level = "待改进"
                report_content.append(f"      • 性能等级: {performance_level}")
                report_content.append("")
        
        # 特征重要性分析
        if hasattr(self, 'shap_values'):
            report_content.append("🔍 特征重要性分析:")
            top_features = self.shap_values['feature_importance'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report_content.append(f"   {i:2d}. {row['Feature']:20s}: {row['Importance']:.6f}")
            report_content.append("")
        
        # 系统配置信息
        report_content.append("⚙️ 系统配置:")
        report_content.append(f"   • 目标列: {self.config.get('target_column', 'carbon_price')}")
        report_content.append(f"   • 序列长度: {self.config.get('sequence_length', 60)}")
        report_content.append(f"   • 测试集比例: {self.config.get('test_size', 0.2)}")
        report_content.append(f"   • 随机种子: {self.config.get('random_state', 42)}")
        report_content.append("")
        
        # 模型详细配置
        if hasattr(self, 'predictions'):
            report_content.append("🔧 模型配置详情:")
            if 'LSTM' in self.predictions:
                lstm_config = self.config.get('lstm_config', {})
                report_content.append(f"   • LSTM模型:")
                report_content.append(f"     - 隐藏层单元: {lstm_config.get('hidden_units', 50)}")
                report_content.append(f"     - 训练轮数: {lstm_config.get('epochs', 100)}")
                report_content.append(f"     - 批量大小: {lstm_config.get('batch_size', 32)}")
            
            if 'Transformer' in self.predictions:
                transformer_config = self.config.get('transformer_config', {})
                report_content.append(f"   • Transformer模型:")
                report_content.append(f"     - 注意力头数: {transformer_config.get('num_heads', 8)}")
                report_content.append(f"     - 模型维度: {transformer_config.get('d_model', 64)}")
                report_content.append(f"     - 训练轮数: {transformer_config.get('epochs', 50)}")
            report_content.append("")
        
        # 应用建议
        report_content.append("💡 应用建议:")
        report_content.append("   • 定期更新模型以保持预测准确性")
        report_content.append("   • 结合SHAP分析结果理解预测逻辑")
        report_content.append("   • 在重大决策前考虑多个模型的集成结果")
        
        if hasattr(self, 'predictions'):
            best_model = max(self.predictions.keys(), 
                           key=lambda x: self.predictions[x]['R²'])
            best_r2 = self.predictions[best_model]['R²']
            
            if best_r2 > 0.8:
                report_content.append("   • 模型性能优秀，可用于实际预测")
            elif best_r2 > 0.6:
                report_content.append("   • 模型性能良好，建议继续优化")
            else:
                report_content.append("   • 模型性能待提升，建议增加特征或调整模型")
        
        report_content.append("   • 监控重要特征的变化趋势")
        report_content.append("   • 结合领域知识理解模型输出")
        report_content.append("")
        
        # 数据质量评估
        if self.processed_data is not None:
            missing_data = self.processed_data.isnull().sum().sum()
            target_col = self.config.get('target_column', 'carbon_price')
            price_volatility = self.processed_data[target_col].std() if target_col in self.processed_data.columns else 0
            
            report_content.append("🔎 数据质量评估:")
            report_content.append(f"   • 缺失值数量: {missing_data}")
            data_completeness = (1 - missing_data / (len(self.processed_data) * len(self.processed_data.columns))) * 100
            report_content.append(f"   • 数据完整性: {data_completeness:.1f}%")
            report_content.append(f"   • 价格波动率: {price_volatility:.2f}")
            report_content.append("")
        
        # 风险警告
        report_content.append("⚠️ 风险提示:")
        report_content.append("   • 模型预测仅供参考，不构成投资建议")
        report_content.append("   • 碳市场受政策影响较大，存在不确定性")
        report_content.append("   • 建议结合多种信息源进行综合判断")
        report_content.append("   • 模型需要定期重新训练以适应市场变化")
        report_content.append("")
        
        report_content.append("=" * 80)
        report_content.append("报告生成完成")
        report_content.append("=" * 80)
        
        # 保存文本报告
        txt_file = os.path.join(self.output_dirs['txt'], f'{self.run_name}_detailed_report.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
    
    def _generate_runtime_log(self):
        """生成运行时日志"""
        log_content = []
        log_content.append("=" * 80)
        log_content.append("碳价格预测系统运行日志")
        log_content.append("=" * 80)
        log_content.append(f"运行时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("")
        
        # 系统环境信息
        import sys
        import platform
        log_content.append("🖥️ 系统环境:")
        log_content.append(f"   • Python版本: {sys.version.split()[0]}")
        log_content.append(f"   • 操作系统: {platform.system()} {platform.release()}")
        log_content.append(f"   • 处理器架构: {platform.machine()}")
        log_content.append("")
        
        # 依赖库信息
        log_content.append("📦 依赖库版本:")
        try:
            import pandas as pd_version
            log_content.append(f"   • pandas: {pd_version.__version__}")
        except:
            log_content.append("   • pandas: 未知版本")
        
        try:
            import numpy as np_version
            log_content.append(f"   • numpy: {np_version.__version__}")
        except:
            log_content.append("   • numpy: 未知版本")
        
        try:
            import sklearn
            log_content.append(f"   • scikit-learn: {sklearn.__version__}")
        except:
            log_content.append("   • scikit-learn: 未知版本")
        
        try:
            import tensorflow as tf
            log_content.append(f"   • tensorflow: {tf.__version__}")
        except:
            log_content.append("   • tensorflow: 未安装")
        
        try:
            import shap
            log_content.append(f"   • shap: {shap.__version__}")
        except:
            log_content.append("   • shap: 未安装")
        
        log_content.append("")
        
        # 运行配置
        log_content.append("⚙️ 运行配置:")
        for key, value in self.config.items():
            log_content.append(f"   • {key}: {value}")
        log_content.append("")
        
        # 数据信息
        if self.processed_data is not None:
            log_content.append("📊 数据信息:")
            log_content.append(f"   • 数据来源: {getattr(self, 'data_source', '示例数据')}")
            log_content.append(f"   • 数据形状: {self.processed_data.shape}")
            log_content.append(f"   • 特征列表: {list(self.feature_names)}")
            log_content.append(f"   • 目标变量: {self.config.get('target_column', 'carbon_price')}")
            log_content.append("")
        
        # 模型训练信息
        if hasattr(self, 'predictions'):
            log_content.append("🤖 模型训练信息:")
            log_content.append(f"   • 训练模型数量: {len(self.predictions)}")
            log_content.append(f"   • 训练完成模型: {list(self.predictions.keys())}")
            
            best_model = max(self.predictions.keys(), 
                           key=lambda x: self.predictions[x]['R²'])
            log_content.append(f"   • 最佳模型: {best_model}")
            log_content.append(f"   • 最佳R²: {self.predictions[best_model]['R²']:.4f}")
            log_content.append("")
        
        # 文件输出信息
        log_content.append("📁 生成文件:")
        log_content.append("   • carbon_prediction_report.xlsx - Excel报告")
        log_content.append("   • carbon_prediction_detailed_report.txt - 详细文本报告")
        log_content.append("   • carbon_prediction_runtime_log.txt - 运行日志")
        log_content.append("   • carbon_prediction_results/ - 可视化图表目录")
        log_content.append("")
        
        # 运行状态
        log_content.append("✅ 运行状态: 成功完成")
        log_content.append("")
        log_content.append("=" * 80)
        log_content.append("日志记录完成")
        log_content.append("=" * 80)
        
        # 保存运行日志
        log_file = os.path.join(self.output_dirs['txt'], f'{self.run_name}_runtime_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))
    
    def _print_summary_report(self):
        """打印简要报告"""
        print("\n" + "="*80)
        print(" " * 25 + "碳价格预测系统分析报告")
        print("="*80)
        
        # 数据信息
        if self.processed_data is not None:
            print(f"\n📈 数据信息:")
            print(f"   • 数据点数量: {len(self.processed_data):,}")
            print(f"   • 特征数量: {len(self.feature_names)}")
            print(f"   • 时间范围: {self.processed_data.index[0].strftime('%Y-%m-%d')} 到 {self.processed_data.index[-1].strftime('%Y-%m-%d')}")
        
        # 模型性能
        if hasattr(self, 'predictions'):
            print(f"\n🏆 模型性能排名（按R²排序）:")
            sorted_models = sorted(self.predictions.items(), 
                                 key=lambda x: x[1]['R²'], reverse=True)
            
            for i, (model, result) in enumerate(sorted_models, 1):
                print(f"   {i}. {model}:")
                print(f"      • R²: {result['R²']:.4f}")
                print(f"      • RMSE: {result['RMSE']:.4f}")
                print(f"      • MAPE: {result['MAPE']:.2f}%")
                print(f"      • 方向准确率: {result['Direction_Accuracy']:.2f}%")
        
        # 关键特征
        if hasattr(self, 'shap_values'):
            print(f"\n🔍 关键影响因子（按SHAP重要性）:")
            top_features = self.shap_values['feature_importance'].head(5)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")
        
        # 模型建议
        print(f"\n💡 模型应用建议:")
        
        if hasattr(self, 'predictions'):
            best_model = max(self.predictions.keys(), 
                           key=lambda x: self.predictions[x]['R²'])
            best_r2 = self.predictions[best_model]['R²']
            
            if best_r2 > 0.8:
                print("   • 模型性能优秀，可用于实际预测")
            elif best_r2 > 0.6:
                print("   • 模型性能良好，建议继续优化")
            else:
                print("   • 模型性能待提升，建议增加特征或调整模型")
        
        print(f"   • 定期更新模型以保持预测准确性")
        print(f"   • 结合SHAP分析结果理解预测逻辑")
        print(f"   • 在重大决策前考虑多个模型的集成结果")
        
        print("\n" + "="*80)
        print(" " * 30 + "分析完成")
        print("="*80)
    
    def run_complete_analysis(self, data_path=None):
        """
        运行完整的分析流程
        
        一键完成所有分析的便捷方法
        =============================
        
        参数:
            data_path: 你的数据文件路径（可选）
                      如果不提供，系统将使用默认的{DEFAULT_DATA_FILE}文件
        
        完整分析流程包括：
        -------------------
        1. 数据加载和验证
        2. 特征工程和数据预处理
        3. 多模型训练（LSTM、Transformer、随机森林等）
        4. 模型性能评估和对比
        5. SHAP可解释性分析
        6. 可视化图表生成
        7. 详细分析报告输出
        
        使用你的数据进行分析：
        -------------------------
        # 方法1：直接指定数据文件
        # system = CarbonPricePredictionSystem()
        # system.run_complete_analysis('你的数据文件.xlsx')
        
        # 方法2：先配置再分析
        # config = {
        #     'target_column': '你的碳价格列名',
        #     'sequence_length': 60,
        #     'test_size': 0.2
        # }
        # system = CarbonPricePredictionSystem(config=config)
        # system.run_complete_analysis('数据文件.xlsx')
        
        # 方法3：分步执行（更灵活）
        # system = CarbonPricePredictionSystem()
        # system.load_data('你的数据.xlsx')     # 加载数据
        # system.preprocess_data()             # 预处理
        # system.train_models()                # 训练模型
        # system.evaluate_models()             # 评估性能
        # system.perform_shap_analysis()       # SHAP分析
        # system.create_visualizations()       # 生成图表
        # system.generate_report()             # 生成报告
        
        输出文件说明：
        ----------------
        - Excel报告：包含所有数值结果和数据表
        - 详细文本报告：完整的分析结果解读
        - 运行日志：系统配置和运行信息
        - 图表文件：模型性能、预测结果、SHAP分析等可视化
        
        数据要求提醒：
        ----------------
        - 确保数据格式正确（日期索引 + 数值列）
        - 数据量充足（建议1000+个数据点）
        - 包含足够的影响因子（建议8-15个变量）
        - 数据质量良好（无异常值，少量缺失值）
        """
        print("🚀 开始碳价格预测完整分析...\n")
        
        try:
            # 1. 数据源处理：根据项目记忆使用默认文件
            if data_path:
                print(f"📊 使用指定的数据文件: {data_path}")
                self.data_source = data_path
                self.load_data(data_path)
            else:
                # 使用默认的测试数据文件
                default_data_path = DEFAULT_DATA_FILE
                print(f"📊 未指定数据文件，使用默认测试数据: {default_data_path}")
                self.data_source = default_data_path
                
                # 尝试加载默认数据文件，如果不存在则创建示例数据
                try:
                    self.load_data(default_data_path)
                except (FileNotFoundError, IOError):
                    print(f"⚠️ 未找到默认数据文件 {default_data_path}，创建示例数据...")
                    self.create_sample_data(save_path=default_data_path)
                    self.data_source = '示例数据'
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 模型训练
            self.train_models()
            
            # 4. 模型评估
            self.evaluate_models()
            
            # 5. SHAP分析
            self.perform_shap_analysis()
            
            # 6. 创建可视化
            self.create_visualizations()
            
            # 7. 生成报告
            self.generate_report()
            
            print("\n✅ 完整分析流程执行成功！")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    主函数演示
    
    📖 如何使用自己的数据运行系统：
    ================================
    
    🔧 方法1：快速开始（推荐新手）
    ----------------------------
    # 直接替换main()函数中的文件路径
    system = CarbonPricePredictionSystem()
    system.run_complete_analysis('你的数据文件.xlsx')  # 改成你的文件路径
    
    📋 数据准备检查清单：
    --------------------
    ✅ 文件格式：Excel(.xlsx/.xls) 或 CSV(.csv)
    ✅ 第一列：日期（作为索引），格式正确
    ✅ 数据量：至少500行，推荐1000+行
    ✅ 碳价格列：包含目标变量
    ✅ 影响因子：8-15个相关变量
    ✅ 数据质量：无异常值，缺失值<5%
    ✅ 时间连续：按时间顺序排列
    
    🎯 预期输出文件：
    ----------------
    • Excel报告：包含所有分析结果和数据表
    • 详细文本报告：完整的分析结果解读  
    • 运行日志：系统配置和运行信息
    • 图表文件：模型性能、预测结果、SHAP分析等可视化
    """
    print("🌍 " + "="*60)
    print(" " * 20 + "碳价格预测系统")
    print(" " * 15 + "LSTM + Transformer + SHAP 分析")
    print("="*60 + " 🌍")
    
    # 🚀 使用自己数据的示例（取消注释并修改路径）：
    # =====================================================
    # 
    # 方法1：快速开始
    # system = CarbonPricePredictionSystem()
    # system.run_complete_analysis('你的数据文件.xlsx')  # 替换为你的文件路径
    # 
    # 方法2：自定义配置
    # my_config = {
    #     'target_column': '你的碳价格列名',  # 如：'carbon_price', 'price', '碳价格'等
    #     'sequence_length': 60,
    #     'test_size': 0.2
    # }
    # system = CarbonPricePredictionSystem(config=my_config)
    # system.run_complete_analysis('你的数据文件.xlsx')
    #
    # 当前运行示例数据演示：
    # =====================
    
    try:
        # 创建预测系统实例
        system = CarbonPricePredictionSystem()
        
        # 🚀 使用全局变量定义的测试数据文件
        test_data_path = DEFAULT_DATA_FILE
        print(f"📊 正在使用默认测试数据文件: {test_data_path}")
        
        # 运行完整分析（使用指定的测试数据文件）
        success = system.run_complete_analysis(test_data_path)
        
        if success:
            print("\n🎉 程序执行成功！")
            print("📁 生成文件:")
            print("   • carbon_price_sample_data.xlsx - 示例数据")
            print("   • carbon_prediction_report.xlsx - 完整Excel分析报告")
            print("   • carbon_prediction_detailed_report.txt - 详细文本分析报告")
            print("   • carbon_prediction_runtime_log.txt - 系统运行日志")
            print("   • carbon_prediction_results/ - 可视化图表目录")
            print("\n🔧 如何使用你自己的数据:")
            print("   1. 数据格式：Excel(.xlsx)或CSV(.csv)，第一列为日期")
            print("   2. 必需列：碳价格列（列名可为carbon_price、price、碳价格等）")
            print("   3. 推荐列：GDP、工业指数、能源价格等影响因子（8-15个）")
            print("   4. 数据量：建议1000+个数据点，时间跨度3年以上")
            print("\n💻 代码示例:")
            print("   # 基本用法")
            print("   system = CarbonPricePredictionSystem()")
            print("   system.run_complete_analysis('你的数据文件.xlsx')")
            print("\n   # 自定义配置")
            print("   config = {'target_column': '你的碳价格列名'}")
            print("   system = CarbonPricePredictionSystem(config=config)")
            print("   system.run_complete_analysis('数据文件.xlsx')")
        else:
            print("\n⚠️ 程序执行失败，请检查错误信息")
    
    except ImportError as e:
        print(f"\n⚠️ 无法导入完整系统 ({str(e)})，运行基础版本...")
        
        # 运行基础版本
        from carbon_test import SimpleCarbonPrediction
        
        print("\n📊 基础版本演示")
        print("="*40)
        
        # 创建基础版本系统
        basic_system = SimpleCarbonPrediction()
        
        # 运行基础版本分析
        basic_system.run_analysis()
        
        print("\n✅ 基础版本演示成功！")
        
    except Exception as e:
        print(f"\n❌ 程序运行错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
