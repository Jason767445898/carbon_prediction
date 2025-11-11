#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Models Comparison for Coal Price Prediction
包含9种基线模型的完整对比系统
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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)

# ======================== 配置参数 ========================
CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',
    'sequence_length': 45,
    'test_size': 0.2,
    'validation_size': 0.1,
    'epochs': 300,
    'batch_size': 24,
    'learning_rate': 0.001,
}

OUTPUT_DIR = 'outputs/baseline_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== Attention机制 ========================
def create_simple_attention(input_tensor, attention_dim):
    """简单注意力机制"""
    query = layers.Dense(attention_dim, name='attention_query')(input_tensor)
    key = layers.Dense(attention_dim, name='attention_key')(input_tensor)
    value = layers.Dense(attention_dim, name='attention_value')(input_tensor)
    scores = layers.Dot(axes=[2, 2])([query, key])
    scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32)))(scores)
    attention_weights = layers.Softmax(axis=-1, name='attention_weights')(scores)
    context = layers.Dot(axes=[2, 1])([attention_weights, value])
    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(context)
    return context_vector

# ======================== 基线模型类 ========================
class BaselineModels:
    """所有基线模型的统一接口"""
    
    def __init__(self):
        self.data = None
        self.feature_names = []
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.results = []  # 修复：应该是列表而不是字典
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_data(self, file_path):
        """加载数据"""
        print(f"\n📊 加载数据文件: {file_path}")
        self.data = pd.read_stata(file_path)
        
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        
        # 仅使用2017年至2021年6月的数据
        start_date = pd.Timestamp('2017-01-01')
        end_date = pd.Timestamp('2021-06-30')
        self.data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        print(f"✅ 数据加载成功: {self.data.shape}")
        print(f"   时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        return self.data
    
    def create_enhanced_features(self, df, target):
        """创建特征工程"""
        print("   • 创建特征...")
        original_features = ['oil_price', 'log_oil_price', 'log_oil_price_sqr', 'log_carbon_price']
        available_features = [f for f in original_features if f in df.columns]
        df = df[[target] + available_features]
        
        # 添加技术指标
        for window in [15, 20]:
            df[f'{target}_ma{window}'] = df[target].rolling(window=window, min_periods=1).mean()
        
        # MACD
        exp12 = df[target].ewm(span=12, adjust=False).mean()
        exp26 = df[target].ewm(span=26, adjust=False).mean()
        df[f'{target}_macd'] = exp12 - exp26
        df[f'{target}_macd_signal'] = df[f'{target}_macd'].ewm(span=9, adjust=False).mean()
        df[f'{target}_macd_hist'] = df[f'{target}_macd'] - df[f'{target}_macd_signal']
        
        # 动量和EMA
        df[f'{target}_momentum'] = df[target].diff(10)
        df[f'{target}_ema12'] = df[target].ewm(span=12, adjust=False).mean()
        df[f'{target}_ema26'] = df[target].ewm(span=26, adjust=False).mean()
        
        # 其他特征的MA5
        for feature in available_features:
            df[f'{feature}_ma5'] = df[feature].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n🔧 数据预处理...")
        df = self.data.copy()
        target = CONFIG['target_column']
        
        df = df.dropna(axis=1, how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        df = self.create_enhanced_features(df, target)
        df = df.dropna()
        
        self.feature_names = [col for col in df.columns if col != target]
        print(f"✅ 预处理完成: {len(self.feature_names)} 特征, {df.shape}")
        return df
    
    def prepare_ml_data(self, df):
        """准备机器学习数据（非序列）"""
        target = CONFIG['target_column']
        X = df[self.feature_names].values
        y = df[target].values
        
        n = len(X)
        train_size = int(n * (1 - CONFIG['test_size'] - CONFIG['validation_size']))
        val_size = int(n * (1 - CONFIG['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_sequences(self, data, feature_cols, target_col, seq_length):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            seq_X = data[feature_cols].iloc[i:i+seq_length].values
            seq_y = data[target_col].iloc[i+seq_length]
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                X.append(seq_X)
                y.append(seq_y)
        return np.array(X), np.array(y)
    
    def prepare_seq_data(self, df):
        """准备序列数据（LSTM和Attention）"""
        target = CONFIG['target_column']
        seq_len = CONFIG['sequence_length']
        X, y = self.create_sequences(df, self.feature_names, target, seq_len)
        
        n = len(X)
        train_size = int(n * (1 - CONFIG['test_size'] - CONFIG['validation_size']))
        val_size = int(n * (1 - CONFIG['test_size']))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """评估模型性能"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # 方向准确率
        if len(y_true) > 1:
            direction_acc = np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == 
                                  np.sign(y_true[1:] - y_true[:-1])) * 100
        else:
            direction_acc = 0.0
        
        results = {
            'model': model_name,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"   {model_name:25s} | R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}%")
        return results
    
    # ==================== 基线模型 1-2: 规则基线 ====================
    def baseline_mean(self, X_train, y_train, X_test, y_test):
        """基线1: 平均值模型"""
        print("\n[1/9] 训练平均值模型...")
        mean_value = np.mean(y_train)
        y_pred = np.full_like(y_test, mean_value)
        return self.evaluate_model(y_test, y_pred, "1. Mean Baseline")
    
    def baseline_median(self, X_train, y_train, X_test, y_test):
        """基线2: 中位数模型"""
        print("\n[2/9] 训练中位数模型...")
        median_value = np.median(y_train)
        y_pred = np.full_like(y_test, median_value)
        return self.evaluate_model(y_test, y_pred, "2. Median Baseline")
    
    # ==================== 基线模型 3: 决策树 ====================
    def baseline_decision_tree(self, X_train, y_train, X_test, y_test):
        """基线3: 决策树"""
        print("\n[3/8] 训练决策树...")
        model = DecisionTreeRegressor(max_depth=4, random_state=21)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate_model(y_test, y_pred, "3. Decision Tree")
    
    # ==================== 基线模型 4: K-近邻 ====================
    def baseline_knn(self, X_train, y_train, X_test, y_test):
        """基线4: K-近邻"""
        print("\n[4/8] 训练K-近邻...")
        model = KNeighborsRegressor(n_neighbors=20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate_model(y_test, y_pred, "4. K-Nearest Neighbors")
    
    # ==================== 基线模型 5: 随机森林 ====================
    def baseline_random_forest(self, X_train, y_train, X_test, y_test):
        """基线5: 随机森林"""
        print("\n[5/8] 训练随机森林...")
        model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=40, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate_model(y_test, y_pred, "5. Random Forest")
    
    # ==================== 基线模型 6: XGBoost ====================
    def baseline_xgboost(self, X_train, y_train, X_test, y_test):
        """基线6: XGBoost"""
        print("\n[6/8] 训练XGBoost...")
        model = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.01, random_state=20, n_jobs=-1)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        return self.evaluate_model(y_test, y_pred, "6. XGBoost")
    
    # ==================== 基线模型 7: GradientBoosting ====================
    def baseline_gradientboosting(self, X_train, y_train, X_test, y_test):
        """基线7: GradientBoosting"""
        print("\n[7/8] 训练GradientBoosting...")
        model = GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.01, random_state=20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.evaluate_model(y_test, y_pred, "7. GradientBoosting")
    
    # ==================== 基线模型 8: LSTM ====================
    def baseline_lstm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """基线8: LSTM"""
        print("\n[8/8] 训练LSTM...")
        
        # 标准化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # 构建LSTM模型
        inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = layers.LSTM(64, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(32, dropout=0.4, recurrent_dropout=0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(48, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(24, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, min_lr=1e-6, verbose=0)
        ]
        
        model.fit(X_train_scaled, y_train_scaled, 
                 validation_data=(X_val_scaled, y_val_scaled),
                 epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                 callbacks=callbacks, verbose=0)
        
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        return self.evaluate_model(y_test, y_pred, "8. LSTM")
    
    # ==================== 基线模型 9: Attention ====================
    def baseline_attention(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """基线9: Attention Only"""
        print("\n[9/9] 训练Attention模型...")
        
        # 标准化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # 构建Attention模型
        inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        
        # 简单的embedding层
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Attention机制
        attention_out = create_simple_attention(x, attention_dim=48)
        
        # 全连接层
        x = layers.Dense(48, activation='relu')(attention_out)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(24, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, min_lr=1e-6, verbose=0)
        ]
        
        model.fit(X_train_scaled, y_train_scaled,
                 validation_data=(X_val_scaled, y_val_scaled),
                 epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                 callbacks=callbacks, verbose=0)
        
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        return self.evaluate_model(y_test, y_pred, "9. Attention")
    
    def visualize_comparison(self):
        """生成对比可视化"""
        print("\n🎨 生成对比可视化...")
        
        # 1. 性能对比条形图 (增加方向准确率)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        models = [r['model'] for r in self.results]
        r2_scores = [r['R2'] for r in self.results]
        rmse_scores = [r['RMSE'] for r in self.results]
        mae_scores = [r['MAE'] for r in self.results]
        mape_scores = [r['MAPE'] for r in self.results]
        direction_acc_scores = [r['Direction_Accuracy'] for r in self.results]
        
        # R² Score
        axes[0, 0].barh(models, r2_scores, color='skyblue')
        axes[0, 0].set_xlabel('R² Score', fontsize=12)
        axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # RMSE
        axes[0, 1].barh(models, rmse_scores, color='lightcoral')
        axes[0, 1].set_xlabel('RMSE', fontsize=12)
        axes[0, 1].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # MAE
        axes[0, 2].barh(models, mae_scores, color='lightgreen')
        axes[0, 2].set_xlabel('MAE', fontsize=12)
        axes[0, 2].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0, 2].grid(axis='x', alpha=0.3)
        
        # MAPE
        axes[1, 0].barh(models, mape_scores, color='orange')
        axes[1, 0].set_xlabel('MAPE (%)', fontsize=12)
        axes[1, 0].set_title('MAPE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Direction Accuracy (新增)
        axes[1, 1].barh(models, direction_acc_scores, color='mediumpurple')
        axes[1, 1].set_xlabel('Direction Accuracy (%)', fontsize=12)
        axes[1, 1].set_title('Direction Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # 隐藏最后一个子图
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_comparison.png'), dpi=300)
        plt.close()
        
        # 2. 预测对比图（显示所有模型）
        sorted_results = sorted(self.results, key=lambda x: x['R2'], reverse=True)
        all_models = sorted_results  # 显示所有模型
        
        fig, axes = plt.subplots(len(all_models), 1, figsize=(14, 5*len(all_models)))
        if len(all_models) == 1:
            axes = [axes]
        
        for idx, result in enumerate(all_models):
            y_true = result['y_true']
            y_pred = result['y_pred']
            show_points = min(200, len(y_true))
            
            axes[idx].plot(y_true[-show_points:], label='Actual', linewidth=2, alpha=0.8)
            axes[idx].plot(y_pred[-show_points:], label='Predicted', linewidth=2, alpha=0.8)
            axes[idx].set_title(f"{result['model']} | R²={result['R2']:.4f}, RMSE={result['RMSE']:.4f}", 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Time Steps')
            axes[idx].set_ylabel('Coal Price')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_predictions.png'), dpi=300)
        plt.close()
        
        print(f"✅ 可视化已保存到 {OUTPUT_DIR}")
    
    def save_report(self):
        """保存对比报告"""
        print("\n📊 保存对比报告...")
        
        report_path = os.path.join(OUTPUT_DIR, f'{self.run_timestamp}_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(" " * 30 + "煤炭价格预测 - 基线模型对比报告\n")
            f.write("=" * 100 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 模型性能排名 (按R²排序):\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'排名':<6} {'模型':<30} {'R²':<12} {'RMSE':<12} {'MAE':<12} {'MAPE(%)':<12} {'方向准确率(%)':<15}\n")
            f.write("-" * 120 + "\n")
            
            sorted_results = sorted(self.results, key=lambda x: x['R2'], reverse=True)
            for rank, result in enumerate(sorted_results, 1):
                f.write(f"{rank:<6} {result['model']:<30} {result['R2']:<12.4f} "
                       f"{result['RMSE']:<12.4f} {result['MAE']:<12.4f} {result['MAPE']:<12.2f} "
                       f"{result['Direction_Accuracy']:<15.2f}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("📈 最佳模型: " + sorted_results[0]['model'] + "\n")
            f.write(f"   R² = {sorted_results[0]['R2']:.4f}\n")
            f.write(f"   RMSE = {sorted_results[0]['RMSE']:.4f}\n")
            f.write(f"   MAE = {sorted_results[0]['MAE']:.4f}\n")
            f.write(f"   MAPE = {sorted_results[0]['MAPE']:.2f}%\n")
            f.write(f"   方向准确率 = {sorted_results[0]['Direction_Accuracy']:.2f}%\n")
            f.write("=" * 100 + "\n")
        
        print(f"✅ 报告已保存: {report_path}")
    
    def run(self):
        """运行所有基线模型"""
        print("\n" + "="*100)
        print(" " * 35 + "煤炭价格预测 - 基线模型对比")
        print("="*100 + "\n")
        
        # 加载和预处理数据
        self.load_data(CONFIG['data_file'])
        df = self.preprocess_data()
        
        # 准备ML数据（非序列）
        X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml = self.prepare_ml_data(df)
        
        # 准备序列数据（LSTM和Attention）
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = self.prepare_seq_data(df)
        
        print("\n" + "="*100)
        print("开始训练所有基线模型...")
        print("="*100)
        
        # 运行所有模型
        self.results.append(self.baseline_mean(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_median(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_decision_tree(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_knn(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_random_forest(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_xgboost(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_gradientboosting(X_train_ml, y_train_ml, X_test_ml, y_test_ml))
        self.results.append(self.baseline_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq))
        self.results.append(self.baseline_attention(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq))
        
        # 可视化和报告
        self.visualize_comparison()
        self.save_report()
        
        print("\n" + "="*100)
        print("✅ 所有基线模型训练完成!")
        print("="*100 + "\n")

if __name__ == '__main__':
    baseline = BaselineModels()
    baseline.run()
