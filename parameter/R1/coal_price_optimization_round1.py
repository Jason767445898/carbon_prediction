#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤炭价格预测模型参数优化 - 第一轮
基于基线R²=-6.4360的系统性优化探索
生成时间: 2025-11-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# 🔥 第一轮优化配置 - 20个配置组合
OPTIMIZATION_CONFIGS = [
    # 组1: 简化架构 + 强正则化 (配置1-4)
    {
        'name': 'Config_01_Simple_Strong_Reg',
        'sequence_length': 30,
        'lstm_units': [64],
        'attention_dim': 32,
        'dense_units_1': 32,
        'dense_units_2': 16,
        'learning_rate': 0.001,
        'lstm_dropout': 0.4,
        'dropout_rate': 0.6,
        'batch_size': 32,
        'l2_reg': 0.01,
    },
    {
        'name': 'Config_02_Simple_Ultra_Reg',
        'sequence_length': 30,
        'lstm_units': [32],
        'attention_dim': 16,
        'dense_units_1': 16,
        'dense_units_2': 8,
        'learning_rate': 0.001,
        'lstm_dropout': 0.5,
        'dropout_rate': 0.7,
        'batch_size': 32,
        'l2_reg': 0.05,
    },
    {
        'name': 'Config_03_Medium_High_Reg',
        'sequence_length': 40,
        'lstm_units': [96],
        'attention_dim': 48,
        'dense_units_1': 48,
        'dense_units_2': 24,
        'learning_rate': 0.0005,
        'lstm_dropout': 0.4,
        'dropout_rate': 0.6,
        'batch_size': 24,
        'l2_reg': 0.02,
    },
    {
        'name': 'Config_04_Balanced_Strong_Reg',
        'sequence_length': 45,
        'lstm_units': [80],
        'attention_dim': 40,
        'dense_units_1': 40,
        'dense_units_2': 20,
        'learning_rate': 0.0008,
        'lstm_dropout': 0.45,
        'dropout_rate': 0.65,
        'batch_size': 28,
        'l2_reg': 0.015,
    },
    
    # 组2: 双层LSTM探索 (配置5-8)
    {
        'name': 'Config_05_TwoLayer_Conservative',
        'sequence_length': 50,
        'lstm_units': [64, 32],
        'attention_dim': 32,
        'dense_units_1': 32,
        'dense_units_2': 16,
        'learning_rate': 0.0003,
        'lstm_dropout': 0.3,
        'dropout_rate': 0.5,
        'batch_size': 20,
        'l2_reg': 0.008,
    },
    {
        'name': 'Config_06_TwoLayer_Medium',
        'sequence_length': 55,
        'lstm_units': [96, 48],
        'attention_dim': 48,
        'dense_units_1': 48,
        'dense_units_2': 24,
        'learning_rate': 0.0004,
        'lstm_dropout': 0.35,
        'dropout_rate': 0.55,
        'batch_size': 22,
        'l2_reg': 0.01,
    },
    {
        'name': 'Config_07_TwoLayer_Small',
        'sequence_length': 35,
        'lstm_units': [48, 24],
        'attention_dim': 24,
        'dense_units_1': 24,
        'dense_units_2': 12,
        'learning_rate': 0.0005,
        'lstm_dropout': 0.4,
        'dropout_rate': 0.6,
        'batch_size': 24,
        'l2_reg': 0.012,
    },
    {
        'name': 'Config_08_TwoLayer_Aggressive',
        'sequence_length': 60,
        'lstm_units': [80, 40],
        'attention_dim': 40,
        'dense_units_1': 40,
        'dense_units_2': 20,
        'learning_rate': 0.0002,
        'lstm_dropout': 0.25,
        'dropout_rate': 0.45,
        'batch_size': 18,
        'l2_reg': 0.005,
    },
    
    # 组3: 序列长度优化 (配置9-12)
    {
        'name': 'Config_09_ShortSeq_HighCap',
        'sequence_length': 20,
        'lstm_units': [128],
        'attention_dim': 64,
        'dense_units_1': 64,
        'dense_units_2': 32,
        'learning_rate': 0.001,
        'lstm_dropout': 0.3,
        'dropout_rate': 0.5,
        'batch_size': 32,
        'l2_reg': 0.005,
    },
    {
        'name': 'Config_10_MediumSeq_Balanced',
        'sequence_length': 40,
        'lstm_units': [64],
        'attention_dim': 32,
        'dense_units_1': 32,
        'dense_units_2': 16,
        'learning_rate': 0.0006,
        'lstm_dropout': 0.35,
        'dropout_rate': 0.55,
        'batch_size': 26,
        'l2_reg': 0.008,
    },
    {
        'name': 'Config_11_LongSeq_Conservative',
        'sequence_length': 70,
        'lstm_units': [48],
        'attention_dim': 24,
        'dense_units_1': 24,
        'dense_units_2': 12,
        'learning_rate': 0.0003,
        'lstm_dropout': 0.4,
        'dropout_rate': 0.6,
        'batch_size': 16,
        'l2_reg': 0.015,
    },
    {
        'name': 'Config_12_VeryLongSeq_Simple',
        'sequence_length': 90,
        'lstm_units': [32],
        'attention_dim': 16,
        'dense_units_1': 16,
        'dense_units_2': 8,
        'learning_rate': 0.0002,
        'lstm_dropout': 0.5,
        'dropout_rate': 0.7,
        'batch_size': 12,
        'l2_reg': 0.02,
    },
    
    # 组4: 学习率与批量优化 (配置13-16)
    {
        'name': 'Config_13_LowLR_SmallBatch',
        'sequence_length': 50,
        'lstm_units': [64],
        'attention_dim': 32,
        'dense_units_1': 32,
        'dense_units_2': 16,
        'learning_rate': 0.0001,
        'lstm_dropout': 0.3,
        'dropout_rate': 0.5,
        'batch_size': 8,
        'l2_reg': 0.005,
    },
    {
        'name': 'Config_14_MediumLR_MediumBatch',
        'sequence_length': 50,
        'lstm_units': [80],
        'attention_dim': 40,
        'dense_units_1': 40,
        'dense_units_2': 20,
        'learning_rate': 0.0007,
        'lstm_dropout': 0.35,
        'dropout_rate': 0.55,
        'batch_size': 20,
        'l2_reg': 0.008,
    },
    {
        'name': 'Config_15_HighLR_LargeBatch',
        'sequence_length': 50,
        'lstm_units': [96],
        'attention_dim': 48,
        'dense_units_1': 48,
        'dense_units_2': 24,
        'learning_rate': 0.002,
        'lstm_dropout': 0.4,
        'dropout_rate': 0.6,
        'batch_size': 48,
        'l2_reg': 0.01,
    },
    {
        'name': 'Config_16_AdaptiveLR_DynamicBatch',
        'sequence_length': 50,
        'lstm_units': [72],
        'attention_dim': 36,
        'dense_units_1': 36,
        'dense_units_2': 18,
        'learning_rate': 0.0004,
        'lstm_dropout': 0.38,
        'dropout_rate': 0.58,
        'batch_size': 16,
        'l2_reg': 0.007,
    },
    
    # 组5: 注意力机制优化 (配置17-20)
    {
        'name': 'Config_17_LargeAttention',
        'sequence_length': 50,
        'lstm_units': [64],
        'attention_dim': 96,
        'dense_units_1': 48,
        'dense_units_2': 24,
        'learning_rate': 0.0005,
        'lstm_dropout': 0.3,
        'dropout_rate': 0.5,
        'batch_size': 24,
        'l2_reg': 0.006,
    },
    {
        'name': 'Config_18_SmallAttention',
        'sequence_length': 50,
        'lstm_units': [96],
        'attention_dim': 24,
        'dense_units_1': 48,
        'dense_units_2': 24,
        'learning_rate': 0.0005,
        'lstm_dropout': 0.35,
        'dropout_rate': 0.55,
        'batch_size': 24,
        'l2_reg': 0.008,
    },
    {
        'name': 'Config_19_BalancedAttention',
        'sequence_length': 50,
        'lstm_units': [80],
        'attention_dim': 56,
        'dense_units_1': 56,
        'dense_units_2': 28,
        'learning_rate': 0.0006,
        'lstm_dropout': 0.32,
        'dropout_rate': 0.52,
        'batch_size': 22,
        'l2_reg': 0.007,
    },
    {
        'name': 'Config_20_MinimalAttention',
        'sequence_length': 50,
        'lstm_units': [112],
        'attention_dim': 16,
        'dense_units_1': 56,
        'dense_units_2': 28,
        'learning_rate': 0.0008,
        'lstm_dropout': 0.38,
        'dropout_rate': 0.58,
        'batch_size': 28,
        'l2_reg': 0.009,
    },
]

BASE_CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 300,
    'lstm_recurrent_dropout': 0.2,
    'scaler_type': 'minmax',
    'use_residual': True,
}

OUTPUT_DIR = 'outputs/optimization_round1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_simple_attention(input_tensor, attention_dim):
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    return context_vector

def build_lstm_attention(sequence_length, n_features, config):
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    lstm_units_list = config['lstm_units'] if isinstance(config['lstm_units'], list) else [config['lstm_units']]
    x = inputs
    
    for i, units in enumerate(lstm_units_list):
        x = layers.LSTM(
            units,
            return_sequences=True,
            dropout=config['lstm_dropout'],
            recurrent_dropout=BASE_CONFIG['lstm_recurrent_dropout'],
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']),
            name=f'lstm_layer_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
    
    lstm_out = x
    attention_out = create_simple_attention(lstm_out, config['attention_dim'])
    
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    if lstm_pooled.shape[-1] != config['attention_dim']:
        lstm_pooled = layers.Dense(config['attention_dim'], name='residual_projection')(lstm_pooled)
    
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    dense = layers.Dense(config['dense_units_1'], activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(combined)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(config['dropout_rate'])(dense)
    
    dense = layers.Dense(config['dense_units_2'], activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(dense)
    dense = layers.Dropout(config['dropout_rate'] * 0.5)(dense)
    
    outputs = layers.Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate'], clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    return model

class CoalPriceOptimizer:
    def __init__(self, config, output_dir=None):
        self.config = {**BASE_CONFIG, **config}
        self.output_dir = output_dir
        self.data = None
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = []
        self.rf_model = None
        self.shap_values = None
        
    def load_data(self, file_path):
        self.data = pd.read_stata(file_path)
        
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        self.data = self.data[(self.data.index.year >= 2017) & (self.data.index.year <= 2024)]
        
        # 仅排除最极端异常时段: 2021.10 到 2022.2（价格暴涨期）
        exclude_start = pd.Timestamp('2021-10-01')
        exclude_end = pd.Timestamp('2022-02-28')
        exclude_condition = (self.data.index >= exclude_start) & (self.data.index <= exclude_end)
        
        self.data = self.data[~exclude_condition]
        return self.data
    
    def create_enhanced_features(self, df, target):
        original_features = [
            'oil_price',
            'log_oil_price',
            'log_oil_price_sqr',
            'log_carbon_price',
        ]
        available_features = [f for f in original_features if f in df.columns]
        df = df[[target] + available_features]
        
        for window in [2, 5, 10]:
            df[f'{target}_ma{window}'] = df[target].rolling(window=window, min_periods=1).mean()
        
        exp12 = df[target].ewm(span=12, adjust=False).mean()
        exp26 = df[target].ewm(span=26, adjust=False).mean()
        df[f'{target}_macd'] = exp12 - exp26
        df[f'{target}_macd_signal'] = df[f'{target}_macd'].ewm(span=9, adjust=False).mean()
        df[f'{target}_macd_hist'] = df[f'{target}_macd'] - df[f'{target}_macd_signal']
        
        delta = df[target].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df[f'{target}_rsi'] = 100 - (100 / (1 + rs))
        
        rolling_mean = df[target].rolling(window=20, min_periods=1).mean()
        rolling_std = df[target].rolling(window=20, min_periods=1).std()
        df[f'{target}_bb_upper'] = rolling_mean + (rolling_std * 2)
        df[f'{target}_bb_lower'] = rolling_mean - (rolling_std * 2)
        df[f'{target}_bb_width'] = (df[f'{target}_bb_upper'] - df[f'{target}_bb_lower']) / rolling_mean
        
        high_14 = df[target].rolling(window=14, min_periods=1).max()
        low_14 = df[target].rolling(window=14, min_periods=1).min()
        df[f'{target}_williams_r'] = -100 * ((high_14 - df[target]) / (high_14 - low_14 + 1e-10))
        
        df[f'{target}_momentum'] = df[target].diff(10)
        df[f'{target}_roc'] = ((df[target] - df[target].shift(12)) / (df[target].shift(12) + 1e-10)) * 100
        df[f'{target}_ema12'] = df[target].ewm(span=12, adjust=False).mean()
        df[f'{target}_ema26'] = df[target].ewm(span=26, adjust=False).mean()
        df[f'{target}_std20'] = df[target].rolling(window=20, min_periods=1).std()
        
        for feature in available_features:
            ma_col_name = f'{feature}_ma5'
            df[ma_col_name] = df[feature].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def preprocess_data(self):
        df = self.data.copy()
        target = self.config['target_column']
        
        df = df.dropna(axis=1, how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        df = self.create_enhanced_features(df, target)
        df = df.dropna()
        self.feature_names = [col for col in df.columns if col != target]
        return df
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        target = self.config['target_column']
        seq_len = self.config['sequence_length']
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        
        n = len(X)
        train_size = int(n * (1 - self.config['test_size'] - self.config['validation_size']))
        val_size = int(n * (1 - self.config['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler_X.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(X_train.shape)
        
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_flat = self.scaler_X.transform(X_val_flat)
        X_val = X_val_flat.reshape(X_val.shape)
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_flat = self.scaler_X.transform(X_test_flat)
        X_test = X_test_flat.reshape(X_test.shape)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model = build_lstm_attention(
            sequence_length=self.config['sequence_length'],
            n_features=X_train.shape[2],
            config=self.config
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, 
                         verbose=0, min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, 
                             min_lr=1e-6, verbose=0)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        return self.model
    
    def evaluate(self, X_test, y_test):
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
        }
    
    def perform_shap_analysis(self, X_train_ml, y_train_ml, X_test_ml):
        """执行SHAP分析"""
        if not SHAP_AVAILABLE:
            return None
        
        # 训练随机森林模型用于SHAP分析
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train_ml, y_train_ml)
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X_test_ml[:100])
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        self.shap_values = {
            'values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'X_test_sample': X_test_ml[:100]
        }
        return self.shap_values
    
    def visualize_and_save(self, results):
        """生成并保存可视化图表和报告"""
        if self.output_dir is None:
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 训练历史图（Loss和MAE）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss', fontsize=12)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE', fontsize=12)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 预测对比图
        y_true = results['y_true']
        y_pred = results['y_pred']
        show_points = min(300, len(y_true))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(y_true[-show_points:], label='Actual', linewidth=2)
        ax.plot(y_pred[-show_points:], label='Predicted', linewidth=2)
        ax.set_title(f'Coal Price Prediction (Last {show_points} Points)', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual Price', fontsize=12)
        ax.set_ylabel('Predicted Price', fontsize=12)
        ax.set_title('Prediction Scatter Plot', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. SHAP总结图（如果可用）
        if self.shap_values is not None and SHAP_AVAILABLE:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values['values'], self.shap_values['X_test_sample'],
                            feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. SHAP条形图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values['values'], self.shap_values['X_test_sample'],
                            feature_names=self.feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. 生成文本报告
        lstm_desc = 'x'.join(map(str, self.config['lstm_units'])) if isinstance(self.config['lstm_units'], list) else str(self.config['lstm_units'])
        report_path = os.path.join(self.output_dir, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("煤炭价格预测模型 - 训练报告\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("模型配置\n")
            f.write("-"*80 + "\n")
            f.write(f"序列长度: {self.config['sequence_length']}\n")
            f.write(f"LSTM层: [{lstm_desc}]\n")
            f.write(f"Attention维度: {self.config['attention_dim']}\n")
            f.write(f"Dense层: [{self.config['dense_units_1']}, {self.config['dense_units_2']}]\n")
            f.write(f"学习率: {self.config['learning_rate']}\n")
            f.write(f"批量大小: {self.config['batch_size']}\n")
            f.write(f"LSTM Dropout: {self.config['lstm_dropout']}\n")
            f.write(f"Dropout: {self.config['dropout_rate']}\n")
            f.write(f"L2正则化: {self.config['l2_reg']}\n\n")
            
            f.write("模型性能\n")
            f.write("-"*80 + "\n")
            f.write(f"R² Score: {results['R2']:.6f}\n")
            f.write(f"RMSE: {results['RMSE']:.4f}\n")
            f.write(f"MAE: {results['MAE']:.4f}\n")
            f.write(f"MAPE: {results['MAPE']:.2f}%\n\n")
            
            f.write("训练信息\n")
            f.write("-"*80 + "\n")
            f.write(f"训练轮数: {len(self.history.history['loss'])}\n")
            f.write(f"最终训练Loss: {self.history.history['loss'][-1]:.6f}\n")
            f.write(f"最终验证Loss: {self.history.history['val_loss'][-1]:.6f}\n")
            f.write(f"最终训练MAE: {self.history.history['mae'][-1]:.6f}\n")
            f.write(f"最终验证MAE: {self.history.history['val_mae'][-1]:.6f}\n\n")
            
            f.write("="*80 + "\n")
    
    def run(self):
        self.load_data(self.config['data_file'])
        df = self.preprocess_data()
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)
        self.train(X_train, y_train, X_val, y_val)
        results = self.evaluate(X_test, y_test)
        
        # 执行SHAP分析
        if SHAP_AVAILABLE:
            target = self.config['target_column']
            n = len(df)
            train_size = int(n * (1 - self.config['test_size']))
            X_train_ml = np.nan_to_num(df[self.feature_names].iloc[:train_size].values)
            y_train_ml = np.nan_to_num(df[target].iloc[:train_size].values)
            X_test_ml = np.nan_to_num(df[self.feature_names].iloc[train_size:].values)
            self.perform_shap_analysis(X_train_ml, y_train_ml, X_test_ml)
        
        self.visualize_and_save(results)
        return results

def run_optimization():
    print("="*80)
    print(" " * 20 + "煤炭价格预测 - 第一轮参数优化")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置数量: {len(OPTIMIZATION_CONFIGS)}")
    print("="*80 + "\n")
    
    results_list = []
    
    for idx, opt_config in enumerate(OPTIMIZATION_CONFIGS, 1):
        config_name = opt_config['name']
        print(f"\n[{idx}/{len(OPTIMIZATION_CONFIGS)}] 运行配置: {config_name}")
        print("-" * 60)
        
        lstm_desc = 'x'.join(map(str, opt_config['lstm_units'])) if isinstance(opt_config['lstm_units'], list) else str(opt_config['lstm_units'])
        print(f"  参数: Seq={opt_config['sequence_length']}, LSTM=[{lstm_desc}], "
              f"Attn={opt_config['attention_dim']}, LR={opt_config['learning_rate']}, "
              f"Batch={opt_config['batch_size']}, L2={opt_config['l2_reg']}")
        
        # 创建配置专属文件夹
        config_folder = os.path.join('parameter', f'config_{idx:02d}')
        os.makedirs(config_folder, exist_ok=True)
        
        try:
            start_time = datetime.now()
            optimizer = CoalPriceOptimizer(opt_config, output_dir=config_folder)
            results = optimizer.run()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # 移除y_true和y_pred，避免DataFrame存储问题
            results_summary = {
                'config_name': config_name,
                'R2': results['R2'],
                'RMSE': results['RMSE'],
                'MAE': results['MAE'],
                'MAPE': results['MAPE'],
                'elapsed_time': elapsed_time,
                'config': opt_config.copy()
            }
            results_list.append(results_summary)
            
            print(f"  ✅ 完成 | R²={results['R2']:.4f}, RMSE={results['RMSE']:.4f}, "
                  f"MAE={results['MAE']:.4f}, MAPE={results['MAPE']:.2f}% | 用时: {elapsed_time:.1f}s")
            print(f"     结果已保存至: {config_folder}")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            results_list.append({
                'config_name': config_name,
                'R2': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'elapsed_time': 0,
                'config': opt_config.copy(),
                'error': str(e)
            })
    
    # 保存结果
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values('R2', ascending=False)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(OUTPUT_DIR, f'{timestamp}_optimization_results.xlsx')
    df_results.to_excel(excel_path, index=False)
    
    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, f'{timestamp}_optimization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("煤炭价格预测模型 - 第一轮参数优化报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置总数: {len(OPTIMIZATION_CONFIGS)}\n")
        f.write(f"成功运行: {df_results['R2'].notna().sum()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Top 10 配置（按R²排序）\n")
        f.write("="*80 + "\n")
        for idx, row in df_results.head(10).iterrows():
            f.write(f"\n{row['config_name']}:\n")
            f.write(f"  R² = {row['R2']:.4f}, RMSE = {row['RMSE']:.4f}, "
                   f"MAE = {row['MAE']:.4f}, MAPE = {row['MAPE']:.2f}%\n")
            if 'config' in row and isinstance(row['config'], dict):
                cfg = row['config']
                lstm_desc = 'x'.join(map(str, cfg['lstm_units'])) if isinstance(cfg['lstm_units'], list) else str(cfg['lstm_units'])
                f.write(f"  参数: Seq={cfg['sequence_length']}, LSTM=[{lstm_desc}], "
                       f"Attn={cfg['attention_dim']}, LR={cfg['learning_rate']}, "
                       f"Batch={cfg['batch_size']}, L2={cfg['l2_reg']}\n")
    
    print("\n" + "="*80)
    print("优化完成!")
    print(f"结果已保存至: {excel_path}")
    print(f"报告已保存至: {report_path}")
    print("="*80)
    
    # 打印Top 5
    print("\n📊 Top 5 配置:")
    print("-" * 80)
    for idx, row in df_results.head(5).iterrows():
        print(f"{row['config_name']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}")
    print("-" * 80)

if __name__ == '__main__':
    run_optimization()
