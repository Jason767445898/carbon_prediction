#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
煤炭价格预测模型 - 第5轮参数优化
基于第4轮最优结果 (R²=0.8228) 进行精细调优
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 第5轮优化配置 - 在最优Config_15基础上精细调优
OPTIMIZATION_CONFIGS = [
    # 组1: 围绕最优Config_15的微调 (Seq=30, LSTM=124, Attn=78)
    {'id': 1, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 26, 'l2': 0.0085},  # 最优基准
    {'id': 2, 'seq': 30, 'lstm': [128], 'attn': 80, 'lr': 0.0012, 'batch': 26, 'l2': 0.0085},  # 略增容量
    {'id': 3, 'seq': 30, 'lstm': [120], 'attn': 76, 'lr': 0.0012, 'batch': 26, 'l2': 0.009},   # 略减容量+强正则
    {'id': 4, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0011, 'batch': 26, 'l2': 0.0085},  # 降低学习率
    {'id': 5, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0013, 'batch': 26, 'l2': 0.008},   # 提高学习率
    
    # 组2: 序列长度29-31的精细探索
    {'id': 6, 'seq': 29, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 26, 'l2': 0.0085},
    {'id': 7, 'seq': 31, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 26, 'l2': 0.0085},
    {'id': 8, 'seq': 29, 'lstm': [120], 'attn': 76, 'lr': 0.0013, 'batch': 26, 'l2': 0.0085},
    {'id': 9, 'seq': 31, 'lstm': [128], 'attn': 80, 'lr': 0.0011, 'batch': 26, 'l2': 0.008},
    
    # 组3: 探索Config_07高性能区 (Seq=32, LSTM=104)
    {'id': 10, 'seq': 32, 'lstm': [104], 'attn': 68, 'lr': 0.0014, 'batch': 26, 'l2': 0.0075},  # Config_07复现
    {'id': 11, 'seq': 32, 'lstm': [108], 'attn': 70, 'lr': 0.0013, 'batch': 26, 'l2': 0.008},
    {'id': 12, 'seq': 32, 'lstm': [100], 'attn': 66, 'lr': 0.0015, 'batch': 26, 'l2': 0.0075},
    
    # 组4: Batch size微调
    {'id': 13, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 24, 'l2': 0.0085},
    {'id': 14, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 28, 'l2': 0.0085},
    {'id': 15, 'seq': 30, 'lstm': [128], 'attn': 80, 'lr': 0.0011, 'batch': 28, 'l2': 0.008},
    
    # 组5: L2正则化精细调优
    {'id': 16, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 26, 'l2': 0.009},
    {'id': 17, 'seq': 30, 'lstm': [124], 'attn': 78, 'lr': 0.0012, 'batch': 26, 'l2': 0.0095},
    {'id': 18, 'seq': 31, 'lstm': [120], 'attn': 76, 'lr': 0.0013, 'batch': 26, 'l2': 0.009},
    
    # 组6: 组合最优要素的新探索
    {'id': 19, 'seq': 29, 'lstm': [128], 'attn': 80, 'lr': 0.0012, 'batch': 28, 'l2': 0.0085},
    {'id': 20, 'seq': 31, 'lstm': [120], 'attn': 76, 'lr': 0.0012, 'batch': 24, 'l2': 0.009},
]

BASE_CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 300,
    'lstm_dropout': 0.4,
    'lstm_recurrent_dropout': 0.3,
    'dropout_rate': 0.5,
    'dense_units_1': 48,
    'dense_units_2': 24,
}

OUTPUT_BASE_DIR = 'parameter/R5'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

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

def build_model(sequence_length, n_features, config):
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    lstm_units_list = config['lstm']
    x = inputs
    
    for i, units in enumerate(lstm_units_list):
        x = layers.LSTM(
            units,
            return_sequences=True,
            dropout=BASE_CONFIG['lstm_dropout'],
            recurrent_dropout=BASE_CONFIG['lstm_recurrent_dropout'],
            name=f'lstm_layer_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
    
    lstm_out = x
    attention_out = create_simple_attention(lstm_out, config['attn'])
    
    lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_pooling')(lstm_out)
    if lstm_pooled.shape[-1] != config['attn']:
        lstm_pooled = layers.Dense(config['attn'], name='residual_projection')(lstm_pooled)
    
    combined = layers.Add(name='residual_connection')([lstm_pooled, attention_out])
    combined = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')(combined)
    
    dense = layers.Dense(BASE_CONFIG['dense_units_1'], activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(config['l2']))(combined)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(BASE_CONFIG['dropout_rate'])(dense)
    
    dense = layers.Dense(BASE_CONFIG['dense_units_2'], activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(config['l2']))(dense)
    dense = layers.Dropout(BASE_CONFIG['dropout_rate'] * 0.6)(dense)
    
    outputs = layers.Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=config['lr'], clipnorm=1.0),
        loss='mse',
        metrics=['mae']
    )
    return model

class CoalPriceOptimizer:
    def __init__(self):
        self.data = None
        self.feature_names = []
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_data(self, file_path):
        print(f"📊 加载数据: {file_path}")
        self.data = pd.read_stata(file_path)
        
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        start_date = pd.Timestamp('2017-01-01')
        end_date = pd.Timestamp('2021-06-30')
        self.data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        print(f"✅ 数据范围: {self.data.index[0]} 到 {self.data.index[-1]}, 形状: {self.data.shape}")
        return self.data
    
    def create_enhanced_features(self, df, target):
        original_features = ['oil_price', 'log_oil_price', 'log_oil_price_sqr', 'log_carbon_price']
        available_features = [f for f in original_features if f in df.columns]
        df = df[[target] + available_features]
        
        # coal_price技术指标
        for window in [2, 5, 10]:
            df[f'{target}_ma{window}'] = df[target].rolling(window=window, min_periods=1).mean()
        
        exp12 = df[target].ewm(span=12, adjust=False).mean()
        exp26 = df[target].ewm(span=26, adjust=False).mean()
        df[f'{target}_macd'] = exp12 - exp26
        df[f'{target}_macd_signal'] = df[f'{target}_macd'].ewm(span=9, adjust=False).mean()
        df[f'{target}_macd_hist'] = df[f'{target}_macd'] - df[f'{target}_macd_signal']
        df[f'{target}_momentum'] = df[target].diff(10)
        df[f'{target}_ema12'] = df[target].ewm(span=12, adjust=False).mean()
        df[f'{target}_ema26'] = df[target].ewm(span=26, adjust=False).mean()
        
        # 其他特征MA5
        for feature in available_features:
            df[f'{feature}_ma5'] = df[feature].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def preprocess_data(self):
        df = self.data.copy()
        target = BASE_CONFIG['target_column']
        
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
    
    def prepare_data(self, df, seq_length):
        target = BASE_CONFIG['target_column']
        X, y = self.create_sequences(df, self.feature_names, target, seq_length)
        
        n = len(X)
        train_size = int(n * (1 - BASE_CONFIG['test_size'] - BASE_CONFIG['validation_size']))
        val_size = int(n * (1 - BASE_CONFIG['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        X_train = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_and_evaluate(self, config):
        config_id = f"round5_config_{config['id']:02d}"
        print(f"\n{'='*80}")
        print(f"🚀 训练配置 {config_id}")
        print(f"{'='*80}")
        print(f"参数: Seq={config['seq']}, LSTM={config['lstm']}, Attn={config['attn']}, "
              f"LR={config['lr']}, Batch={config['batch']}, L2={config['l2']}")
        
        try:
            np.random.seed(42)
            tf.random.set_seed(42)
            
            df = self.preprocess_data()
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df, config['seq'])
            
            model = build_model(config['seq'], X_train.shape[2], config)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, 
                            verbose=0, min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, 
                                min_lr=1e-6, verbose=0)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=BASE_CONFIG['epochs'],
                batch_size=config['batch'],
                callbacks=callbacks,
                verbose=0
            )
            
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
            y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            direction_acc = np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == 
                                   np.sign(y_true[1:] - y_true[:-1])) * 100
            
            print(f"✅ R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, "
                  f"MAPE={mape:.2f}%, 方向准确率={direction_acc:.2f}%")
            
            # 保存结果
            output_dir = os.path.join(OUTPUT_BASE_DIR, config_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成可视化
            self.save_visualizations(history, y_true, y_pred, output_dir, config_id)
            
            # 保存报告
            report_path = os.path.join(output_dir, 'report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"配置 {config_id} - 性能报告\n")
                f.write("="*80 + "\n")
                f.write(f"R² Score: {r2:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"MAE: {mae:.4f}\n")
                f.write(f"MAPE: {mape:.2f}%\n")
                f.write(f"方向准确率: {direction_acc:.2f}%\n")
                f.write("\n参数配置:\n")
                f.write(f"  序列长度: {config['seq']}\n")
                f.write(f"  LSTM单元: {config['lstm']}\n")
                f.write(f"  Attention维度: {config['attn']}\n")
                f.write(f"  学习率: {config['lr']}\n")
                f.write(f"  Batch大小: {config['batch']}\n")
                f.write(f"  L2正则化: {config['l2']}\n")
            
            return {
                'config_id': config_id,
                'config': config,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'direction_acc': direction_acc
            }
            
        except Exception as e:
            print(f"❌ 配置 {config_id} 训练失败: {str(e)}")
            return None
    
    def save_visualizations(self, history, y_true, y_pred, output_dir, config_id):
        # 训练历史
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{config_id} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title(f'{config_id} - MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_id}_training.png'), dpi=300)
        plt.close()
        
        # 预测对比
        fig, ax = plt.subplots(figsize=(14, 6))
        show_points = min(300, len(y_true))
        ax.plot(y_true[-show_points:], label='Actual', linewidth=2)
        ax.plot(y_pred[-show_points:], label='Predicted', linewidth=2)
        ax.set_title(f'{config_id} - Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_id}_predictions.png'), dpi=300)
        plt.close()
        
        # 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_title(f'{config_id} - Scatter Plot')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_id}_scatter.png'), dpi=300)
        plt.close()
    
    def run_optimization(self):
        print("\n" + "="*80)
        print(" "*25 + "第5轮参数优化")
        print("="*80)
        print(f"配置数量: {len(OPTIMIZATION_CONFIGS)}")
        print(f"优化策略: 在第4轮最优结果(R²=0.8228)基础上精细调优")
        print("="*80 + "\n")
        
        self.load_data(BASE_CONFIG['data_file'])
        
        results = []
        for i, config in enumerate(OPTIMIZATION_CONFIGS, 1):
            print(f"\n进度: [{i}/{len(OPTIMIZATION_CONFIGS)}]")
            result = self.train_and_evaluate(config)
            if result:
                results.append(result)
        
        # 生成汇总报告
        self.generate_summary_report(results)
        
        print("\n" + "="*80)
        print("✅ 第5轮优化完成!")
        print("="*80 + "\n")
    
    def generate_summary_report(self, results):
        print("\n" + "="*80)
        print("📊 生成汇总报告...")
        print("="*80)
        
        results_sorted = sorted(results, key=lambda x: x['r2'], reverse=True)
        
        summary_dir = os.path.join(OUTPUT_BASE_DIR, 'optimization_round5')
        os.makedirs(summary_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(summary_dir, f'{timestamp}_optimization_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("煤炭价格预测模型 - 第5轮参数优化报告\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置总数: {len(OPTIMIZATION_CONFIGS)}\n")
            f.write(f"成功运行: {len(results)}\n")
            f.write(f"基准性能: R²=0.8228 (第4轮最优)\n")
            f.write("\n" + "="*80 + "\n")
            f.write(f"Top 10 配置（按R²排序）\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results_sorted[:10], 1):
                config = result['config']
                lstm_str = 'x'.join(map(str, config['lstm']))
                config_name = f"Config_{config['id']:02d}_LSTM{lstm_str}_Seq{config['seq']}"
                
                f.write(f"{config_name}:\n")
                f.write(f"  R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}, "
                       f"MAE = {result['mae']:.4f}, MAPE = {result['mape']:.2f}%, "
                       f"方向准确率 = {result['direction_acc']:.2f}%\n")
                f.write(f"  参数: Seq={config['seq']}, LSTM={config['lstm']}, "
                       f"Attn={config['attn']}, LR={config['lr']}, "
                       f"Batch={config['batch']}, L2={config['l2']}\n\n")
        
        print(f"✅ 汇总报告已保存: {report_path}")
        print("\n📊 Top 5 配置:")
        for i, result in enumerate(results_sorted[:5], 1):
            print(f"  {i}. {result['config_id']}: R²={result['r2']:.4f}, "
                  f"RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")

if __name__ == '__main__':
    optimizer = CoalPriceOptimizer()
    optimizer.run_optimization()
