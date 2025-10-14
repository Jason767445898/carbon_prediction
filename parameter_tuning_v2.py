#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数调优脚本 v2.0 - 使用改进的数据预处理
用于调整LSTM和Transformer模型的超参数
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/Jason/Desktop/code/AI')

from carbon_price_prediction import CarbonPricePredictionSystem

def create_tuning_log():
    """创建参数调优日志文件"""
    log_content = [
        "=" * 70,
        "碳价格预测模型参数调优记录 v2.0（改进数据预处理）",
        "=" * 70,
        f"调优开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "🔧 数据预处理改进:",
        "- 自动移除全为NaN的列（var9, var10）",
        "- 使用多层插值方法：线性、多项式、样条、时间序列",
        "- 避免使用0填充，提高数据质量",
        "",
        "🎯 调优目标:",
        "- LSTM: 在R²=0.6778基础上继续提升至R²>0.75",
        "- Transformer: 解决过拟合问题，至少达到R²>0",
        "- 找到最优的超参数配置",
        "",
        "=" * 70,
        ""
    ]
    
    log_file = 'parameter_tuning_v2.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    
    print(f"已创建参数调优日志文件: {log_file}")
    return log_file

def log_tuning_result(log_file, config, results, notes=""):
    """记录调优结果"""
    log_entry = [
        f"调优时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "配置参数:",
        f"  LSTM配置: {config.get('lstm_config', 'N/A')}",
        f"  Transformer配置: {config.get('transformer_config', 'N/A')}",
        "",
        "模型性能:",
    ]
    
    for model_name, metrics in results.items():
        log_entry.append(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ['predictions', 'actual']:
                log_entry.append(f"    {metric}: {value:.4f}")
    
    if notes:
        log_entry.append(f"\n备注: {notes}")
    
    log_entry.append("-" * 70)
    log_entry.append("")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_entry))

def tune_lstm_parameters(log_file):
    """调优LSTM模型参数"""
    print("\n" + "=" * 70)
    print("🔍 开始LSTM模型参数调优")
    print("=" * 70)
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'transformer_config': {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.3,
            'epochs': 50
        }
    }
    
    # LSTM参数组合 - 基于batch_size=16的最佳结果优化
    lstm_configs = [
        # 1. 最佳基线配置
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16,
            'name': '基线配置(batch_size=16)'
        },
        # 2. 更小batch_size
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 8,
            'name': '更小batch_size=8'
        },
        # 3. 降低dropout
        {
            'units': [64, 32],
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 16,
            'name': '降低dropout至0.1'
        },
        # 4. 增加网络宽度
        {
            'units': [128, 64],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16,
            'name': '增加网络宽度'
        },
        # 5. 增加网络深度
        {
            'units': [96, 64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16,
            'name': '增加网络深度'
        },
        # 6. 更多训练轮数
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 16,
            'name': '增加训练轮数至150'
        },
        # 7. 组合优化：小batch + 低dropout
        {
            'units': [64, 32],
            'dropout': 0.15,
            'epochs': 120,
            'batch_size': 8,
            'name': '小batch+低dropout'
        },
        # 8. 组合优化：宽网络 + 小batch
        {
            'units': [128, 64],
            'dropout': 0.15,
            'epochs': 100,
            'batch_size': 8,
            'name': '宽网络+小batch'
        }
    ]
    
    best_lstm_r2 = -float('inf')
    best_lstm_config = None
    best_lstm_results = None
    
    for i, lstm_config in enumerate(lstm_configs):
        config_name = lstm_config.pop('name')
        print(f"\n📊 测试LSTM配置 {i+1}/{len(lstm_configs)}: {config_name}")
        print(f"   参数: {lstm_config}")
        
        config = base_config.copy()
        config['lstm_config'] = lstm_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(log_file, config, results, f"LSTM配置 {i+1}: {config_name}")
            
            # 检查LSTM模型性能
            if 'lstm' in results:
                lstm_r2 = results['lstm']['R²']
                print(f"   ✅ R² = {lstm_r2:.4f}, RMSE = {results['lstm']['RMSE']:.2f}")
                
                if lstm_r2 > best_lstm_r2:
                    best_lstm_r2 = lstm_r2
                    best_lstm_config = lstm_config.copy()
                    best_lstm_config['name'] = config_name
                    best_lstm_results = results['lstm'].copy()
                    print(f"   🏆 新的最佳配置！")
            else:
                print(f"   ❌ LSTM模型训练失败")
                    
        except Exception as e:
            error_msg = f"LSTM配置 {i+1} ({config_name}) 失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            log_tuning_result(log_file, config, {}, error_msg)
    
    return best_lstm_config, best_lstm_results

def tune_transformer_parameters(log_file):
    """调优Transformer模型参数"""
    print("\n" + "=" * 70)
    print("🔍 开始Transformer模型参数调优")
    print("=" * 70)
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'lstm_config': {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        }
    }
    
    # Transformer参数组合 - 简化模型防止过拟合
    transformer_configs = [
        # 1. 超轻量级配置
        {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dff': 128,
            'dropout': 0.3,
            'epochs': 80,
            'name': '超轻量级(1层)'
        },
        # 2. 轻量级配置
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.3,
            'epochs': 50,
            'name': '轻量级(2层)'
        },
        # 3. 高正则化
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.5,
            'epochs': 50,
            'name': '高dropout(0.5)'
        },
        # 4. 平衡配置
        {
            'd_model': 96,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 384,
            'dropout': 0.3,
            'epochs': 60,
            'name': '平衡配置'
        },
        # 5. 更多训练轮数
        {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dff': 256,
            'dropout': 0.3,
            'epochs': 100,
            'name': '更多epochs(100)'
        }
    ]
    
    best_transformer_r2 = -float('inf')
    best_transformer_config = None
    best_transformer_results = None
    
    for i, transformer_config in enumerate(transformer_configs):
        config_name = transformer_config.pop('name')
        print(f"\n📊 测试Transformer配置 {i+1}/{len(transformer_configs)}: {config_name}")
        print(f"   参数: {transformer_config}")
        
        config = base_config.copy()
        config['transformer_config'] = transformer_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(log_file, config, results, f"Transformer配置 {i+1}: {config_name}")
            
            # 检查Transformer模型性能
            if 'transformer' in results:
                transformer_r2 = results['transformer']['R²']
                print(f"   ✅ R² = {transformer_r2:.4f}, RMSE = {results['transformer']['RMSE']:.2f}")
                
                if transformer_r2 > best_transformer_r2:
                    best_transformer_r2 = transformer_r2
                    best_transformer_config = transformer_config.copy()
                    best_transformer_config['name'] = config_name
                    best_transformer_results = results['transformer'].copy()
                    print(f"   🏆 新的最佳配置！")
            else:
                print(f"   ❌ Transformer模型训练失败")
                    
        except Exception as e:
            error_msg = f"Transformer配置 {i+1} ({config_name}) 失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            log_tuning_result(log_file, config, {}, error_msg)
    
    return best_transformer_config, best_transformer_results

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 碳价格预测模型参数调优 v2.0")
    print("=" * 70)
    print("\n🔧 数据预处理改进:")
    print("   ✅ 移除全为NaN的无效列")
    print("   ✅ 使用多层插值方法替代0填充")
    print("   ✅ 提高数据质量和模型性能")
    
    # 创建调优日志
    log_file = create_tuning_log()
    
    # 调优LSTM参数
    best_lstm_config, best_lstm_results = tune_lstm_parameters(log_file)
    
    # 调优Transformer参数
    best_transformer_config, best_transformer_results = tune_transformer_parameters(log_file)
    
    # 记录最佳配置
    print("\n" + "=" * 70)
    print("🏆 参数调优完成 - 最佳配置")
    print("=" * 70)
    
    final_log = [
        "\n" + "=" * 70,
        "最终调优结果（使用改进的数据预处理）",
        "=" * 70,
        ""
    ]
    
    if best_lstm_config and best_lstm_results:
        print(f"\n📊 最佳LSTM配置: {best_lstm_config.get('name', 'N/A')}")
        print(f"   配置: {best_lstm_config}")
        print(f"   R² = {best_lstm_results['R²']:.4f}")
        print(f"   RMSE = {best_lstm_results['RMSE']:.2f}")
        print(f"   MAE = {best_lstm_results['MAE']:.2f}")
        print(f"   MAPE = {best_lstm_results['MAPE']:.2f}%")
        
        final_log.extend([
            f"最佳LSTM配置: {best_lstm_config}",
            f"  R² = {best_lstm_results['R²']:.4f}",
            f"  RMSE = {best_lstm_results['RMSE']:.2f}",
            f"  MAE = {best_lstm_results['MAE']:.2f}",
            f"  MAPE = {best_lstm_results['MAPE']:.2f}%",
            ""
        ])
    else:
        print("\n❌ 未找到有效的LSTM配置")
        final_log.append("最佳LSTM配置: 未找到\n")
    
    if best_transformer_config and best_transformer_results:
        print(f"\n📊 最佳Transformer配置: {best_transformer_config.get('name', 'N/A')}")
        print(f"   配置: {best_transformer_config}")
        print(f"   R² = {best_transformer_results['R²']:.4f}")
        print(f"   RMSE = {best_transformer_results['RMSE']:.2f}")
        print(f"   MAE = {best_transformer_results['MAE']:.2f}")
        print(f"   MAPE = {best_transformer_results['MAPE']:.2f}%")
        
        final_log.extend([
            f"最佳Transformer配置: {best_transformer_config}",
            f"  R² = {best_transformer_results['R²']:.4f}",
            f"  RMSE = {best_transformer_results['RMSE']:.2f}",
            f"  MAE = {best_transformer_results['MAE']:.2f}",
            f"  MAPE = {best_transformer_results['MAPE']:.2f}%",
            ""
        ])
    else:
        print("\n❌ 未找到有效的Transformer配置")
        final_log.append("最佳Transformer配置: 未找到\n")
    
    final_log.extend([
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70
    ])
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("\n" + "=" * 70)
    print(f"📄 详细记录已保存至: {log_file}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
