#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤炭价格预测 - 第一轮参数优化
目标: 针对coal_price进行参数寻优
基准: 沿用第八轮优化方案的架构,调整参数适配煤炭价格特性
"""

import sys
import os
sys.path.append('/Users/Jason/Desktop/code/AI')

# 参数搜索空间
PARAM_GRID = [
    # 配置1: 基准配置 (第八轮架构 + coal_price)
    {
        'name': 'Config1_Baseline',
        'sequence_length': 60,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 80,
        'attention_dim': 160,
        'dropout_rate': 0.35,
        'l2_reg': 0.0005,
        'learning_rate': 0.0002,
        'epochs': 300,
        'batch_size': 32,
    },
    
    # 配置2: 增强序列长度 (煤炭价格趋势性强)
    {
        'name': 'Config2_LongerSeq',
        'sequence_length': 90,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 80,
        'attention_dim': 160,
        'dropout_rate': 0.35,
        'l2_reg': 0.0005,
        'learning_rate': 0.0002,
        'epochs': 300,
        'batch_size': 32,
    },
    
    # 配置3: 更大模型容量
    {
        'name': 'Config3_BiggerModel',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 192,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 300,
        'batch_size': 32,
    },
    
    # 配置4: 更快学习率
    {
        'name': 'Config4_HigherLR',
        'sequence_length': 60,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 80,
        'attention_dim': 160,
        'dropout_rate': 0.35,
        'l2_reg': 0.0005,
        'learning_rate': 0.0003,
        'epochs': 300,
        'batch_size': 32,
    },
    
    # 配置5: 小批次 + 更多轮次
    {
        'name': 'Config5_SmallBatch',
        'sequence_length': 60,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 80,
        'attention_dim': 160,
        'dropout_rate': 0.35,
        'l2_reg': 0.0005,
        'learning_rate': 0.0002,
        'epochs': 400,
        'batch_size': 16,
    },
    
    # 配置6: 更强正则化 (防止过拟合)
    {
        'name': 'Config6_HighReg',
        'sequence_length': 60,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 80,
        'attention_dim': 160,
        'dropout_rate': 0.45,
        'l2_reg': 0.002,
        'learning_rate': 0.0002,
        'epochs': 300,
        'batch_size': 32,
    },
]


def run_optimization():
    """执行参数优化"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # 导入主系统
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lstm_model",
        "/Users/Jason/Desktop/code/AI/lstm_attention_carbon_prediction.py"
    )
    lstm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_module)
    
    results = []
    
    print("\n" + "="*80)
    print(" " * 20 + "煤炭价格预测 - 第一轮参数优化")
    print("="*80 + "\n")
    print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数配置数量: {len(PARAM_GRID)}")
    print(f"目标列: coal_price")
    print("\n" + "="*80 + "\n")
    
    for idx, params in enumerate(PARAM_GRID, 1):
        print(f"\n{'='*80}")
        print(f"执行配置 {idx}/{len(PARAM_GRID)}: {params['name']}")
        print(f"{'='*80}\n")
        
        # 更新配置
        lstm_module.CONFIG.update(params)
        lstm_module.CONFIG['target_column'] = 'coal_price'  # 确保目标列正确
        
        print("📋 当前配置:")
        for key, value in params.items():
            if key != 'name':
                print(f"   • {key}: {value}")
        print()
        
        try:
            # 创建系统实例
            system = lstm_module.LSTMAttentionCarbonPrediction()
            
            # 加载数据
            system.load_data('/Users/Jason/Desktop/code/AI/data.dta')
            
            # 数据预处理
            system.preprocess_data()
            
            # 分割数据
            X_train, y_train, X_val, y_val, X_test, y_test = system.split_and_scale_data()
            
            # 训练模型
            system.train_model(X_train, y_train, X_val, y_val)
            
            # 评估模型
            metrics = system.evaluate_model(X_test, y_test)
            
            # 记录结果
            result = {
                'config_name': params['name'],
                **params,
                **metrics,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'best_epoch': len(system.history.history['loss']) - 
                             lstm_module.CONFIG.get('patience', 60),
            }
            results.append(result)
            
            print(f"\n✅ {params['name']} 完成!")
            print(f"   • R² = {metrics['R2']:.4f}")
            print(f"   • RMSE = {metrics['RMSE']:.4f}")
            print(f"   • MAPE = {metrics['MAPE']:.2f}%")
            print(f"   • 方向准确率 = {metrics['Direction_Accuracy']:.2f}%")
            
        except Exception as e:
            print(f"\n❌ {params['name']} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result = {
                'config_name': params['name'],
                **params,
                'MSE': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan,
                'R2': np.nan,
                'MAPE': np.nan,
                'Direction_Accuracy': np.nan,
                'error': str(e),
            }
            results.append(result)
    
    # 保存结果
    print("\n" + "="*80)
    print("优化完成 - 汇总结果")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(results)
    
    # 按R²排序
    results_df = results_df.sort_values('R2', ascending=False)
    
    # 保存到Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/Users/Jason/Desktop/code/AI/parameter/coal_price_optimization_round1_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 完整结果
        results_df.to_excel(writer, sheet_name='完整结果', index=False)
        
        # Top 3 配置
        top3 = results_df.head(3)[['config_name', 'R2', 'RMSE', 'MAPE', 
                                    'Direction_Accuracy', 'sequence_length',
                                    'lstm_units', 'learning_rate', 'batch_size']]
        top3.to_excel(writer, sheet_name='Top3配置', index=False)
        
        # 参数对比
        param_cols = ['config_name', 'sequence_length', 'lstm_units', 'lstm_units_2',
                     'lstm_units_3', 'attention_dim', 'dropout_rate', 'l2_reg',
                     'learning_rate', 'batch_size', 'epochs']
        params_df = results_df[param_cols]
        params_df.to_excel(writer, sheet_name='参数配置', index=False)
        
        # 性能指标对比
        metrics_cols = ['config_name', 'MSE', 'MAE', 'RMSE', 'R2', 'MAPE', 
                       'Direction_Accuracy']
        metrics_df = results_df[metrics_cols]
        metrics_df.to_excel(writer, sheet_name='性能指标', index=False)
    
    print(f"✅ 结果已保存到: {output_path}\n")
    
    # 打印Top 3
    print("🏆 Top 3 配置:\n")
    for idx, row in results_df.head(3).iterrows():
        print(f"{row['config_name']}:")
        print(f"   • R² = {row['R2']:.4f}")
        print(f"   • RMSE = {row['RMSE']:.4f}")
        print(f"   • MAPE = {row['MAPE']:.2f}%")
        print(f"   • 方向准确率 = {row['Direction_Accuracy']:.2f}%")
        print()
    
    print("="*80 + "\n")
    
    return results_df


if __name__ == '__main__':
    results = run_optimization()
