#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的单层 LSTM + Attention 煤炭价格预测模型
使用 data.dta 数据文件预测 coal_price
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# SHAP分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available, skipping interpretability analysis")

warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 配置参数 - 简化配置
# ============================================================================

CONFIG = {
    # ========== 数据配置 ==========
    'data_file': 'data.dta',           # Stata数据文件路径
    'target_column': 'coal_price',     # 预测目标列：煤炭价格
    
    # ========== 序列与数据划分 ==========
    'sequence_length': 20,             # 🔥 优化：适度增加序列长度（15→20，捕捉更多时间依赖）
    'test_size': 0.2,                  # 测试集比例：20%
    'validation_size': 0.1,            # 验证集比例：10%
    
    # ========== 训练超参数（精细优化） ==========
    'epochs': 400,                     # 🔥 优化：增加训练轮次（给予更多学习时间）
    'batch_size': 16,                  # 🔥 优化：减小批次（更细粒度更新）
    'learning_rate': 0.0005,           # 🔥 优化：降低学习率（更稳定收敛）
    'use_lr_scheduler': True,          # 启用学习率衰减
    'lr_patience': 20,                 # 🔥 优化：更大patience（避免过早衰减）
    'lr_factor': 0.6,                  # 🔥 优化：更温和衰减（保留学习能力）
    'use_gradient_clip': True,         # 🔥 启用梯度裁剪
    'gradient_clip_value': 1.0,        # 🔥 梯度裁剪阈值
    
    # ========== 模型架构参数（平衡优化） ==========
    'num_lstm_layers': 2,              # 🔥 优化：改为双层LSTM（增强特征提取）
    'lstm_units': [80, 48],            # 🔥 优化：适度增加单元数（递减架构）
    'attention_dim': 48,               # 🔥 优化：增加attention维度（32→48）
    'lstm_dropout': 0.25,              # 🔥 优化：降低dropout（允许更多学习）
    'lstm_recurrent_dropout': 0.15,    # 🔥 优化：降低recurrent dropout
    'dropout_rate': 0.35,              # 🔥 优化：降低全连接dropout
    'dense_units_1': 48,               # 🔥 优化：增加第一层（增强表达能力）
    'dense_units_2': 24,               # 🔥 优化：增加第二层
    'use_l2_reg': True,                # 启用L2正则化
    'l2_lambda': 0.0008,               # 🔥 优化：适度L2正则化
    
    # ========== 数据处理参数（优化特征工程 - 仅MA5） ==========
    'scaler_type': 'standard',         # 🔥 优化：改用StandardScaler（更适合LSTM）
    'remove_outliers': True,           # 🔥 保持异常值移除
    'outlier_threshold': 4.5,          # 🔥 优化：稍微放宽阈值（保留更多数据）
    'feature_selection': True,         # 🔥 启用特征选择
    'top_features': 15,                # 🔥 优化：增加到15个特征（10→15，平衡复杂度和信息量）
    'data_augmentation': True,         # 🔥 优化：启用轻微数据增强（增加样本多样性）
    'augmentation_noise': 0.005,       # 🔥 优化：降低噪声（避免过度扰动）
    'augmentation_ratio': 0.15,        # 🔥 优化：降低增强比例（15%，避免过多噪声）
}

# 输出目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 模型构建
# ============================================================================

def create_simple_attention(input_tensor, attention_dim):
    """
    创建简单的 Attention 层
    
    参数:
        input_tensor: shape=(batch_size, sequence_length, features)
        attention_dim: attention维度
    
    返回:
        context_vector: shape=(batch_size, attention_dim)
    """
    # Query, Key, Value 转换
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    
    # 计算 attention scores
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    
    # Softmax 获得 attention weights
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    
    # 应用 attention weights
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    
    # 全局平均池化
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    
    return context_vector


def build_simple_lstm_attention(sequence_length, n_features):
    """
    构建双层 LSTM + Attention 模型（带残差连接）
    激进优化版：双层LSTM + 更强特征提取
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # 🔥 双层 LSTM（递减架构）
    num_layers = CONFIG.get('num_lstm_layers', 2)
    lstm_units_list = CONFIG.get('lstm_units', [96, 64])
    
    # 确保 lstm_units_list 是列表
    if not isinstance(lstm_units_list, list):
        lstm_units_list = [lstm_units_list]
    
    lstm_out = inputs
    for i in range(num_layers):
        units = lstm_units_list[i] if i < len(lstm_units_list) else lstm_units_list[-1]
        return_sequences = (i < num_layers - 1) or True  # 最后一层也返回序列（给Attention使用）
        
        if CONFIG.get('use_l2_reg', False):
            from tensorflow.keras import regularizers
            lstm_out = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=CONFIG.get('lstm_dropout', 0.2),
                recurrent_dropout=CONFIG.get('lstm_recurrent_dropout', 0.1),
                kernel_regularizer=regularizers.l2(CONFIG.get('l2_lambda', 0.0005)),
                recurrent_regularizer=regularizers.l2(CONFIG.get('l2_lambda', 0.0005)),
                name=f'lstm_layer_{i+1}'
            )(lstm_out)
        else:
            lstm_out = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=CONFIG.get('lstm_dropout', 0.2),
                recurrent_dropout=CONFIG.get('lstm_recurrent_dropout', 0.1),
                name=f'lstm_layer_{i+1}'
            )(lstm_out)
        
        lstm_out = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(lstm_out)
    
    # Attention 层
    attention_out = create_simple_attention(lstm_out, CONFIG['attention_dim'])
    
    # 残差连接：将LSTM聚合表示与attention输出相加
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    
    # 维度匹配：最后LSTM层与attention维度对齐
    final_lstm_units = lstm_units_list[-1] if isinstance(lstm_units_list, list) else lstm_units_list
    if CONFIG['attention_dim'] != final_lstm_units:
        lstm_pooled = layers.Dense(CONFIG['attention_dim'], name='residual_projection')(lstm_pooled)
    
    # 残差连接：Add层
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    
    # Layer Normalization
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    # 全连接层（优化结构）
    dense_units_1 = CONFIG.get('dense_units_1', 96)
    dense_units_2 = CONFIG.get('dense_units_2', 48)
    
    if CONFIG.get('use_l2_reg', False):
        from tensorflow.keras import regularizers
        dense = layers.Dense(
            dense_units_1, 
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.get('l2_lambda', 0.0005))
        )(combined)
    else:
        dense = layers.Dense(dense_units_1, activation='relu')(combined)
    
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(CONFIG['dropout_rate'])(dense)
    
    if CONFIG.get('use_l2_reg', False):
        from tensorflow.keras import regularizers
        dense = layers.Dense(
            dense_units_2, 
            activation='relu',
            kernel_regularizer=regularizers.l2(CONFIG.get('l2_lambda', 0.0005))
        )(dense)
    else:
        dense = layers.Dense(dense_units_2, activation='relu')(dense)
    
    dense = layers.Dropout(CONFIG['dropout_rate'] * 0.6)(dense)
    
    # 输出层
    outputs = layers.Dense(1)(dense)
    
    # 编译模型（激进优化）
    optimizer = Adam(learning_rate=CONFIG['learning_rate'])
    
    # 🔥 如果启用梯度裁剪，设置裁剪值
    if CONFIG.get('use_gradient_clip', False):
        optimizer = Adam(
            learning_rate=CONFIG['learning_rate'],
            clipvalue=CONFIG.get('gradient_clip_value', 1.0)
        )
        print(f"   • 启用梯度裁剪: {CONFIG.get('gradient_clip_value', 1.0)}")
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================================================
# 数据处理
# ============================================================================

class SimpleCoalPricePrediction:
    """简单的煤炭价格预测系统"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.history = None
        self.scaler_X = None  # 将在prepare_data中初始化
        self.scaler_y = None  # 将在prepare_data中初始化
        self.feature_names = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.shap_values = None
        self.rf_model = None
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"📊 加载数据文件: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取 Stata 文件
        self.data = pd.read_stata(file_path)
        
        # 转换日期列
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        # 🔥 筛选2017-2024年的数据，但排除2021年6月到2022年1月
        original_shape = self.data.shape
        
        # 保留2017-2024年的数据
        self.data = self.data[(self.data.index.year >= 2017) & (self.data.index.year <= 2024)]
        
        # 排除2021年6月到2022年1月的数据
        exclude_start = pd.Timestamp('2021-06-01')
        exclude_end = pd.Timestamp('2022-01-31')
        exclude_condition = (self.data.index >= exclude_start) & (self.data.index <= exclude_end)
        self.data = self.data[~exclude_condition]
        
        print(f"✅ 数据加载成功")
        print(f"   • 原始数据形状: {original_shape}")
        print(f"   • 筛选后数据形状: {self.data.shape}")
        print(f"   • 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        print(f"   • 筛选条件: 2017-2024年，排除2021年6月到2022年1月")
        
        return self.data
    
    def create_enhanced_features(self, df, target):
        """使用原始数据列作为特征 + 添加所有特征的移动平均（MA）"""
        print("   • 使用原始数据列作为特征 + 添加移动平均（MA）...")
        
        # 🎯 定义要使用的原始特征列（已移除log_coal_price、log_coal_price_sqr，以及指定排除的特征）
        original_features = [
            'oil_price',                        # 石油价格
            'gas_price',                        # 天然气价格
            'carbon_price_hb_ea',               # 碳价格(湖北)
            # 'transactionamount_hb_ea',        # 交易量(湖北) - 已排除
            # 'aqi_hb',                         # 空气质量指数 - 已排除
            # 'highest_temperature',            # 最高温度 - 已排除
            'log_oil_price',                    # 对数石油价格
            'log_gas_price',                    # 对数天然气价格
            'log_carbon_price_hb_ea',           # 对数碳价格
            # 'log_transactionamount_hb_ea',    # 对数交易量 - 已排除
            # 'log_aqi_hb',                     # 对数空气质量指数 - 已排除（因为aqi_hb被排除）
            # 'log_highest_temperature',        # 对数最高温度 - 已排除（因为highest_temperature被排除）
            'log_oil_price_sqr',                # 对数石油价格平方
            'log_gas_price_sqr',                # 对数天然气价格平方
            # 'log_transactionamount_hb_ea_sqr', # 对数交易量平方 - 已排除
            # 'log_aqi_hb_sqr',                 # 对数空气质量指数平方 - 已排除（因为aqi_hb被排除）
        ]
        
        # 打印可用的原始特征
        available_features = [f for f in original_features if f in df.columns]
        print(f"      ✅ 找到 {len(available_features)}/{len(original_features)} 个原始特征")
        print(f"      特征列表: {available_features[:5]}...")
        
        # 只保留目标列和可用的原始特征
        cols_to_keep = [target] + available_features
        df = df[cols_to_keep]
        
        # 🔥 优化建议：只添加MA5移动平均
        print("      • 添加MA5移动平均特征...")
        ma_windows = [5]  # 🔥 只使用5日移动平均
        ma_count = 0
        
        for feature in available_features:
            for window in ma_windows:
                ma_col_name = f'{feature}_ma{window}'
                df[ma_col_name] = df[feature].rolling(window=window, min_periods=1).mean()
                ma_count += 1
        
        # 🔥 新增：为 log_coal_price 添加 MA5（但不包含 log_coal_price 本身）
        if 'log_coal_price' in df.columns:
            print("      • 添加 log_coal_price 的 MA5 特征（不包含 log_coal_price 本身）...")
            df['log_coal_price_ma5'] = df['log_coal_price'].rolling(window=5, min_periods=1).mean()
            ma_count += 1
        
        print(f"      ✅ 已添加 {ma_count} 个MA5移动平均特征（{len(available_features)} 特征 + log_coal_price × 1 窗口）")
        
        return df
    
    def remove_outliers(self, df, target):
        """使用MAD方法移除异常值"""
        if not CONFIG['remove_outliers']:
            return df
        
        print("   • 移除异常值...")
        original_len = len(df)
        
        # 对目标变量使用MAD方法
        median = df[target].median()
        mad = np.median(np.abs(df[target] - median))
        threshold = CONFIG['outlier_threshold']
        
        # 计算修正的z-score
        modified_z_scores = 0.6745 * (df[target] - median) / (mad + 1e-10)
        
        # 移除异常值
        df = df[np.abs(modified_z_scores) < threshold]
        
        removed = original_len - len(df)
        print(f"      移除了 {removed} 个异常值 ({removed/original_len*100:.2f}%)")
        
        return df
    
    def select_features(self, df, target):
        """基于互信息的特征选择"""
        if not CONFIG['feature_selection']:
            return df
        
        print("   • 执行特征选择...")
        
        feature_cols = [col for col in df.columns if col != target]
        
        if len(feature_cols) <= CONFIG['top_features']:
            print(f"      特征数量 ({len(feature_cols)}) <= 阈值 ({CONFIG['top_features']})，保留所有特征")
            return df
        
        # 准备数据
        X = df[feature_cols].fillna(0)
        y = df[target]
        
        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # 选择top特征
        top_features = feature_importance.head(CONFIG['top_features'])['feature'].tolist()
        
        print(f"      从 {len(feature_cols)} 个特征中选择了 {len(top_features)} 个最重要特征")
        print(f"      Top 5 特征: {top_features[:5]}")
        
        # 保留目标列和选中的特征
        selected_cols = [target] + top_features
        
        return df[selected_cols]
    
    def preprocess_data(self):
        """增强的数据预处理流程"""
        print("\n🔧 数据预处理...")
        
        df = self.data.copy()
        target = CONFIG['target_column']
        
        # 检查目标列
        if target not in df.columns:
            raise ValueError(f"目标列 '{target}' 不存在")
        
        print(f"   • 原始形状: {df.shape}")
        
        # 1. 处理缺失值
        print("   • 处理缺失值...")
        df = df.dropna(axis=1, how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 2. 移除无穷大
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        # 3. 创建增强特征
        df = self.create_enhanced_features(df, target)
        
        # 4. 删除包含 NaN 的行
        df = df.dropna()
        
        # 5. 移除异常值
        df = self.remove_outliers(df, target)
        
        # 6. 特征选择
        df = self.select_features(df, target)
        
        # 7. 获取最终特征列
        self.feature_names = [col for col in df.columns if col != target]
        
        print(f"✅ 数据预处理完成")
        print(f"   • 特征数量: {len(self.feature_names)}")
        print(f"   • 数据形状: {df.shape}")
        
        return df
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        """创建序列数据"""
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        
        return np.array(X), np.array(y)
    
    def get_scaler(self, scaler_type):
        """根据类型获取scaler"""
        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            return MinMaxScaler()
    
    def augment_data(self, X, y):
        """🔥 数据增强：添加噪声 + 时间扰动"""
        if not CONFIG['data_augmentation']:
            return X, y
        
        print("   • 应用数据增强...")
        noise_level = CONFIG['augmentation_noise']
        augment_ratio = CONFIG.get('augmentation_ratio', 0.2)
        
        # 计算增强样本数量
        n_augmented = int(len(X) * augment_ratio)
        
        # 随机选择要增强的样本
        indices = np.random.choice(len(X), n_augmented, replace=False)
        
        X_aug_list = []
        y_aug_list = []
        
        for idx in indices:
            # 方法1：高斯噪声
            noise = np.random.normal(0, noise_level, X[idx].shape)
            X_noisy = X[idx] + noise
            X_aug_list.append(X_noisy)
            y_aug_list.append(y[idx])
            
            # 方法2：缩放变换（轻微）
            scale = np.random.uniform(0.98, 1.02)
            X_scaled = X[idx] * scale
            X_aug_list.append(X_scaled)
            y_aug_list.append(y[idx] * scale)
        
        # 转换列表为数组（保持3D维度）
        if len(X_aug_list) > 0:
            X_aug_array = np.array(X_aug_list)  # shape: (n_augmented*2, seq_len, n_features)
            y_aug_array = np.array(y_aug_list)  # shape: (n_augmented*2,)
            
            # 合并原始和增强数据
            X_combined = np.concatenate([X, X_aug_array], axis=0)
            y_combined = np.concatenate([y, y_aug_array], axis=0)
            
            # 打乱顺序
            shuffle_idx = np.random.permutation(len(X_combined))
            X_combined = X_combined[shuffle_idx]
            y_combined = y_combined[shuffle_idx]
            
            print(f"      增强后：{len(X)} → {len(X_combined)} 样本 (+{len(X_aug_list)} 增强样本)")
            
            return X_combined, y_combined
        else:
            return X, y
    
    def prepare_data(self, df):
        """增强的数据准备流程"""
        print("\n📊 准备训练数据...")
        
        target = CONFIG['target_column']
        seq_len = CONFIG['sequence_length']
        
        # 创建序列
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        
        print(f"   • 序列数量: {len(X)}")
        
        # 时间序列分割
        n = len(X)
        train_size = int(n * (1 - CONFIG['test_size'] - CONFIG['validation_size']))
        val_size = int(n * (1 - CONFIG['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        print(f"   • 训练集: {len(X_train)} 样本")
        print(f"   • 验证集: {len(X_val)} 样本")
        print(f"   • 测试集: {len(X_test)} 样本")
        
        # 选择scaler类型
        scaler_type = CONFIG['scaler_type']
        print(f"   • 使用 {scaler_type.upper()} 标准化方法")
        
        self.scaler_X = self.get_scaler(scaler_type)
        self.scaler_y = self.get_scaler(scaler_type)
        
        # 标准化X
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler_X.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_flat = self.scaler_X.transform(X_val_flat)
        X_val = X_val_flat.reshape(X_val.shape)
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = self.scaler_X.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
        
        # 标准化y
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 数据增强（仅训练集）
        X_train, y_train = self.augment_data(X_train, y_train)
        
        print(f"✅ 数据准备完成")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train, y_train, X_val, y_val):
        """🔥 训练模型（激进优化版）"""
        print("\n🤖 训练双层 LSTM + Attention 模型...")
        
        n_features = X_train.shape[2]
        
        # 构建模型
        self.model = build_simple_lstm_attention(
            sequence_length=CONFIG['sequence_length'],
            n_features=n_features
        )
        
        print(f"\n模型架构（🔥 激进优化版）:")
        lstm_units = CONFIG['lstm_units']
        if isinstance(lstm_units, list):
            print(f"   • LSTM层数: {CONFIG.get('num_lstm_layers', 2)}")
            print(f"   • LSTM单元数: {lstm_units}")
        else:
            print(f"   • LSTM单元数: {lstm_units}")
        print(f"   • LSTM Dropout: {CONFIG.get('lstm_dropout', 0.2)}")
        print(f"   • LSTM Recurrent Dropout: {CONFIG.get('lstm_recurrent_dropout', 0.1)}")
        print(f"   • Attention维度: {CONFIG['attention_dim']}")
        print(f"   • 全连接层: [{CONFIG.get('dense_units_1', 96)}, {CONFIG.get('dense_units_2', 48)}]")
        print(f"   • Dropout率: {CONFIG['dropout_rate']}")
        print(f"   • L2正则化: {CONFIG.get('use_l2_reg', False)}")
        if CONFIG.get('use_l2_reg', False):
            print(f"   • L2系数: {CONFIG.get('l2_lambda', 0.0005)}")
        print(f"   • 学习率: {CONFIG['learning_rate']}")
        print(f"   • 批次大小: {CONFIG['batch_size']}")
        print(f"   • 序列长度: {CONFIG['sequence_length']}")
        print(f"   • 残差连接: 已启用")
        if CONFIG.get('use_gradient_clip', False):
            print(f"   • 梯度裁剪: 已启用 (clipvalue={CONFIG.get('gradient_clip_value', 1.0)})")
        self.model.summary()
        
        # 回调函数（优化版）
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=50,
                restore_best_weights=True, 
                verbose=1
            )
        ]
        
        # 添加学习率调度器
        if CONFIG.get('use_lr_scheduler', False):
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=CONFIG.get('lr_factor', 0.3),
                patience=CONFIG.get('lr_patience', 10),
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(lr_scheduler)
            print(f"   • 学习率调度: 已启用 (patience={CONFIG.get('lr_patience', 10)}, factor={CONFIG.get('lr_factor', 0.3)})")
        
        # 训练
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ 模型训练完成")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n📈 评估模型性能...")
        
        # 预测
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # 反标准化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # 方向准确率
        direction_acc = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == 
            np.sign(y_true[1:] - y_true[:-1])
        ) * 100
        
        print(f"\n模型评估结果:")
        print(f"   • R² (决定系数): {r2:.4f}")
        print(f"   • RMSE (均方根误差): {rmse:.4f}")
        print(f"   • MAE (平均绝对误差): {mae:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        print(f"   • 方向准确率: {direction_acc:.2f}%")
        
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc
        }
        
        return results
    
    def perform_shap_analysis(self, X_train_ml, y_train_ml, X_test_ml):
        """执行SHAP可解释性分析"""
        if not SHAP_AVAILABLE:
            print("\n⚠️ SHAP未安装，跳过可解释性分析")
            return None
        
        print("\n🔍 执行SHAP分析...")
        
        # 训练随机森林模型用于SHAP分析
        print("   • 训练随机森林模型...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_ml, y_train_ml)
        
        # 创建SHAP解释器
        print("   • 创建SHAP解释器...")
        explainer = shap.TreeExplainer(self.rf_model)
        
        # 计算SHAP值（使用测试集样本）
        print("   • 计算SHAP值...")
        shap_values = explainer.shap_values(X_test_ml[:100])  # 使用前100个样本
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print(f"\n   Top 10 重要特征:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['Feature']:30s}: {row['Importance']:.6f}")
        
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_test_sample': X_test_ml[:100]
        }
        
        print("\n✅ SHAP分析完成")
        
        return self.shap_values
    
    def visualize(self, results):
        """生成可视化"""
        print("\n🎨 生成可视化图表...")
        
        # 1. 训练历史
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_training.png'), dpi=300)
        plt.close()
        
        # 2. 预测结果
        fig, ax = plt.subplots(figsize=(14, 6))
        
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        show_points = min(300, len(y_true))
        
        ax.plot(y_true[-show_points:], label='Actual', linewidth=2, marker='o', markersize=3)
        ax.plot(y_pred[-show_points:], label='Predicted', linewidth=2, marker='s', markersize=3)
        ax.fill_between(range(show_points), 
                        y_true[-show_points:],
                        y_pred[-show_points:],
                        alpha=0.2, color='gray')
        ax.set_title(f'Coal Price Prediction (Last {show_points} Points)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Coal Price')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        textstr = f'R² = {results["R2"]:.4f}\nRMSE = {results["RMSE"]:.4f}\nMAPE = {results["MAPE"]:.2f}%'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_predictions.png'), dpi=300)
        plt.close()
        
        # 3. 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # 理想预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Coal Price', fontsize=12)
        ax.set_ylabel('Predicted Coal Price', fontsize=12)
        ax.set_title('Prediction Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_scatter.png'), dpi=300)
        plt.close()
        
        # 4. SHAP可视化
        if self.shap_values is not None and SHAP_AVAILABLE:
            print("   • 创建SHAP可视化...")
            
            # SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values['values'], 
                self.shap_values['X_test_sample'],
                feature_names=self.feature_names,
                show=False
            )
            plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_summary.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP Bar Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values['values'],
                self.shap_values['X_test_sample'],
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.title('SHAP Feature Importance (Bar)', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_shap_bar.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature Importance Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = self.shap_values['feature_importance'].head(15)
            ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Top 15 Important Features (SHAP)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_feature_importance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ 可视化图表已保存到: {OUTPUT_DIR}")
    
    def save_report(self, results):
        """保存报告"""
        print("\n📊 保存分析报告...")
        
        report_path = os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_coal_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("简单单层 LSTM + Attention 煤炭价格预测报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 模型配置（参数优化版）:\n")
            f.write(f"   • 目标变量: {CONFIG['target_column']}\n")
            f.write(f"   • 序列长度: {CONFIG['sequence_length']}\n")
            f.write(f"   • LSTM单元数: {CONFIG['lstm_units']}\n")
            f.write(f"   • LSTM Dropout: {CONFIG.get('lstm_dropout', 0.3)}\n")
            f.write(f"   • LSTM Recurrent Dropout: {CONFIG.get('lstm_recurrent_dropout', 0.2)}\n")
            f.write(f"   • Attention维度: {CONFIG['attention_dim']}\n")
            f.write(f"   • 全连接层: [{CONFIG.get('dense_units_1', 128)}, {CONFIG.get('dense_units_2', 64)}]\n")
            f.write(f"   • Dropout率: {CONFIG['dropout_rate']}\n")
            f.write(f"   • L2正则化: {CONFIG.get('use_l2_reg', False)}\n")
            if CONFIG.get('use_l2_reg', False):
                f.write(f"   • L2系数: {CONFIG.get('l2_lambda', 0.001)}\n")
            f.write(f"   • 学习率: {CONFIG['learning_rate']}\n")
            f.write(f"   • 学习率调度: {CONFIG.get('use_lr_scheduler', False)}\n")
            if CONFIG.get('use_lr_scheduler', False):
                f.write(f"   • LR Patience: {CONFIG.get('lr_patience', 15)}\n")
                f.write(f"   • LR Factor: {CONFIG.get('lr_factor', 0.5)}\n")
            f.write(f"   • 批次大小: {CONFIG['batch_size']}\n")
            f.write(f"   • 最大Epochs: {CONFIG['epochs']}\n\n")
            
            f.write("🔧 数据处理配置:\n")
            f.write(f"   • 标准化方法: {CONFIG['scaler_type']}\n")
            f.write(f"   • 异常值移除: {CONFIG['remove_outliers']}\n")
            f.write(f"   • 特征选择: {CONFIG['feature_selection']}\n")
            if CONFIG['feature_selection']:
                f.write(f"   • 保留特征数: {CONFIG['top_features']}\n")
            f.write(f"   • 数据增强: {CONFIG['data_augmentation']}\n\n")
            
            f.write("📈 模型性能:\n")
            f.write(f"   • R² (决定系数): {results['R2']:.4f}\n")
            f.write(f"   • RMSE (均方根误差): {results['RMSE']:.4f}\n")
            f.write(f"   • MAE (平均绝对误差): {results['MAE']:.4f}\n")
            f.write(f"   • MAPE: {results['MAPE']:.2f}%\n")
            f.write(f"   • 方向准确率: {results['Direction_Accuracy']:.2f}%\n\n")
            
            f.write("📁 生成文件:\n")
            f.write(f"   • 训练历史图: {self.run_timestamp}_training.png\n")
            f.write(f"   • 预测对比图: {self.run_timestamp}_predictions.png\n")
            f.write(f"   • 散点分布图: {self.run_timestamp}_scatter.png\n")
            if self.shap_values is not None:
                f.write(f"   • SHAP摘要图: {self.run_timestamp}_shap_summary.png\n")
                f.write(f"   • SHAP条形图: {self.run_timestamp}_shap_bar.png\n")
                f.write(f"   • 特征重要性图: {self.run_timestamp}_feature_importance.png\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✅ 报告已保存到: {report_path}")
    
    def run(self):
        """完整运行流程"""
        print("\n" + "="*80)
        print(" " * 15 + "简单单层 LSTM + Attention 煤炭价格预测系统")
        print("="*80 + "\n")
        
        # 1. 加载数据
        self.load_data(CONFIG['data_file'])
        
        # 2. 预处理
        df = self.preprocess_data()
        
        # 3. 准备数据
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)
        
        # 4. 训练模型
        self.train(X_train, y_train, X_val, y_val)
        
        # 5. 评估模型
        results = self.evaluate(X_test, y_test)
        
        # 6. SHAP分析
        if SHAP_AVAILABLE:
            # 准备用于SHAP的数据（不使用序列）
            target = CONFIG['target_column']
            n = len(df)
            train_size = int(n * (1 - CONFIG['test_size']))
            
            X_train_ml = df[self.feature_names].iloc[:train_size].values
            y_train_ml = df[target].iloc[:train_size].values
            X_test_ml = df[self.feature_names].iloc[train_size:].values
            
            # 处理NaN
            X_train_ml = np.nan_to_num(X_train_ml, nan=0.0)
            y_train_ml = np.nan_to_num(y_train_ml, nan=0.0)
            X_test_ml = np.nan_to_num(X_test_ml, nan=0.0)
            
            self.perform_shap_analysis(X_train_ml, y_train_ml, X_test_ml)
        
        # 7. 可视化
        self.visualize(results)
        
        # 8. 保存报告
        self.save_report(results)
        
        print("\n" + "="*80)
        print("✅ 分析完成!")
        print("="*80 + "\n")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    predictor = SimpleCoalPricePrediction()
    predictor.run()
