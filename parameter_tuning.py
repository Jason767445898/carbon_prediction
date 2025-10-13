#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数调优脚本
用于调整LSTM和Transformer模型的超参数
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/Jason/Desktop/code/AI')

from carbon_price_prediction import CarbonPricePredictionSystem

def create_tuning_log():
    """创建参数调优日志文件"""
    log_content = [
        "==========================================",
        "碳价格预测模型参数调优记录",
        "==========================================",
        f"调优开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "调优目标:",
        "- 提升LSTM和Transformer模型的预测性能",
        "- 优化超参数配置",
        "- 记录调优过程和结果",
        "",
        "==========================================",
        ""
    ]
    
    with open('parameter_tuning.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    
    print("已创建参数调优日志文件: parameter_tuning.txt")

def log_tuning_result(config, results, notes=""):
    """记录调优结果"""
    log_entry = [
        f"调优时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "配置参数:",
        f"  LSTM配置: {config.get('lstm_config', 'N/A')}",
        f"  Transformer配置: {config.get('transformer_config', 'N/A')}",
        "模型性能:",
    ]
    
    for model_name, metrics in results.items():
        log_entry.append(f"  {model_name}:")
        for metric, value in metrics.items():
            if metric not in ['predictions', 'actual']:
                log_entry.append(f"    {metric}: {value:.4f}")
    
    if notes:
        log_entry.append(f"备注: {notes}")
    
    log_entry.append("-" * 50)
    log_entry.append("")
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_entry))
    
    print("已记录调优结果到 parameter_tuning.txt")

def tune_lstm_parameters():
    """调优LSTM模型参数"""
    print("开始调优LSTM模型参数...")
    
    # 基础配置
    base_config = {
        'target_column': 'coal_price',
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.1,
        'transformer_config': {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        }
    }
    
    # LSTM参数组合
    lstm_configs = [
        # 基础配置
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32
        },
        # 增加网络复杂度
        {
            'units': [128, 64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32
        },
        # 调整dropout
        {
            'units': [64, 32],
            'dropout': 0.3,
            'epochs': 100,
            'batch_size': 32
        },
        # 调整批次大小
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 16
        },
        # 增加训练轮数
        {
            'units': [64, 32],
            'dropout': 0.2,
            'epochs': 150,
            'batch_size': 32
        }
    ]
    
    best_lstm_r2 = -float('inf')
    best_lstm_config = None
    best_lstm_results = None
    
    for i, lstm_config in enumerate(lstm_configs):
        print(f"\n测试LSTM配置 {i+1}/{len(lstm_configs)}")
        
        # 创建系统实例
        config = base_config.copy()
        config['lstm_config'] = lstm_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(config, results, f"LSTM配置测试 {i+1}")
            
            # 检查LSTM模型性能
            if 'lstm' in results:
                lstm_r2 = results['lstm']['R²']
                if lstm_r2 > best_lstm_r2:
                    best_lstm_r2 = lstm_r2
                    best_lstm_config = lstm_config.copy()
                    best_lstm_results = results['lstm'].copy()
                    
        except Exception as e:
            error_msg = f"LSTM配置测试 {i+1} 失败: {str(e)}"
            print(error_msg)
            log_tuning_result(config, {}, error_msg)
    
    return best_lstm_config, best_lstm_results

def tune_transformer_parameters():
    """调优Transformer模型参数"""
    print("开始调优Transformer模型参数...")
    
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
            'batch_size': 32
        }
    }
    
    # Transformer参数组合
    transformer_configs = [
        # 基础配置
        {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        },
        # 增加模型维度
        {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        },
        # 增加注意力头数
        {
            'd_model': 128,
            'num_heads': 16,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        },
        # 增加层数
        {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 6,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 50
        },
        # 调整dropout
        {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.2,
            'epochs': 50
        },
        # 增加训练轮数
        {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout': 0.1,
            'epochs': 100
        }
    ]
    
    best_transformer_r2 = -float('inf')
    best_transformer_config = None
    best_transformer_results = None
    
    for i, transformer_config in enumerate(transformer_configs):
        print(f"\n测试Transformer配置 {i+1}/{len(transformer_configs)}")
        
        # 创建系统实例
        config = base_config.copy()
        config['transformer_config'] = transformer_config
        
        try:
            system = CarbonPricePredictionSystem(config=config)
            system.load_data('data.dta')
            system.preprocess_data()
            system.train_models()
            results, _ = system.evaluate_models()
            
            # 记录结果
            log_tuning_result(config, results, f"Transformer配置测试 {i+1}")
            
            # 检查Transformer模型性能
            if 'transformer' in results:
                transformer_r2 = results['transformer']['R²']
                if transformer_r2 > best_transformer_r2:
                    best_transformer_r2 = transformer_r2
                    best_transformer_config = transformer_config.copy()
                    best_transformer_results = results['transformer'].copy()
                    
        except Exception as e:
            error_msg = f"Transformer配置测试 {i+1} 失败: {str(e)}"
            print(error_msg)
            log_tuning_result(config, {}, error_msg)
    
    return best_transformer_config, best_transformer_results

def main():
    """主函数"""
    print("🚀 开始参数调优...")
    
    # 创建调优日志
    create_tuning_log()
    
    # 调优LSTM参数
    print("\n" + "="*60)
    print("LSTM模型参数调优")
    print("="*60)
    best_lstm_config, best_lstm_results = tune_lstm_parameters()
    
    # 调优Transformer参数
    print("\n" + "="*60)
    print("Transformer模型参数调优")
    print("="*60)
    best_transformer_config, best_transformer_results = tune_transformer_parameters()
    
    # 记录最佳配置
    print("\n" + "="*60)
    print("参数调优完成 - 最佳配置")
    print("="*60)
    
    final_log = [
        "最终调优结果:",
        "=============",
        f"最佳LSTM配置: {best_lstm_config}",
        f"  R²: {best_lstm_results['R²']:.4f}" if best_lstm_results else "  R²: N/A",
        f"  RMSE: {best_lstm_results['RMSE']:.4f}" if best_lstm_results else "  RMSE: N/A",
        "",
        f"最佳Transformer配置: {best_transformer_config}",
        f"  R²: {best_transformer_results['R²']:.4f}" if best_transformer_results else "  R²: N/A",
        f"  RMSE: {best_transformer_results['RMSE']:.4f}" if best_transformer_results else "  RMSE: N/A",
        "",
        f"调优结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=========================================="
    ]
    
    with open('parameter_tuning.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(final_log))
    
    print("参数调优已完成，详细记录请查看 parameter_tuning.txt")

if __name__ == "__main__":
    main()