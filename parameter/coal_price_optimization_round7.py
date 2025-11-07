#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤炭价格预测 - 第七轮参数优化
基于第五轮重大突破(R²=0.462, 2层LSTM)和第六轮失败经验(单层LSTM不可行)
核心策略: 回归2层LSTM架构 + 微调学习率、正则化、Attention维度
"""

import sys
import os
sys.path.append('/Users/Jason/Desktop/code/AI')

# 参数搜索空间 - 基于2层LSTM架构的深度优化
PARAM_GRID = [
    # 配置1: 第五轮最优配置 + 低学习率(第六轮发现)
    {
        'name': 'Round7_Config1_2Layer384_LowLR',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.0001,  # 第六轮最优发现
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置2: 更低学习率
    {
        'name': 'Round7_Config2_2Layer384_VeryLowLR',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00008,  # 更低
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置3: 增加dropout（降低过拟合）
    {
        'name': 'Round7_Config3_2Layer384_HighDrop',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.5,  # 0.4→0.5
        'l2_reg': 0.0015,     # 增加L2
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置4: 减少dropout（增加模型容量）
    {
        'name': 'Round7_Config4_2Layer384_LowDrop',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.35,  # 0.4→0.35
        'l2_reg': 0.0008,      # 减少L2
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置5: 增加Attention维度
    {
        'name': 'Round7_Config5_2Layer384_AttDim320',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 320,  # 256→320
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置6: 减少Attention维度
    {
        'name': 'Round7_Config6_2Layer384_AttDim192',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 192,  # 256→192（匹配第二层LSTM）
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置7: 更大2层LSTM（512-256）
    {
        'name': 'Round7_Config7_2Layer512_LowLR',
        'sequence_length': 60,
        'lstm_units': 512,
        'lstm_units_2': 256,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.42,  # 稍增正则
        'l2_reg': 0.0012,
        'learning_rate': 0.0001,  # 低学习率
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置8: 更小2层LSTM（320-160）
    {
        'name': 'Round7_Config8_2Layer320_LowLR',
        'sequence_length': 60,
        'lstm_units': 320,
        'lstm_units_2': 160,
        'lstm_units_3': 0,
        'attention_dim': 192,
        'num_attention_heads': 1,
        'dropout_rate': 0.38,  # 稍减正则
        'l2_reg': 0.0009,
        'learning_rate': 0.0001,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置9: 序列长度45（减少时间依赖）
    {
        'name': 'Round7_Config9_2Layer384_Seq45',
        'sequence_length': 45,  # 60→45
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置10: 序列长度75（增加时间依赖）
    {
        'name': 'Round7_Config10_2Layer384_Seq75',
        'sequence_length': 75,  # 60→75
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置11: 黄金比例LSTM（384-237）
    {
        'name': 'Round7_Config11_2Layer384_Golden',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 237,  # 384 * 0.618 ≈ 237
        'lstm_units_3': 0,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置12: 最优组合（低LR + 高dropout + 大Attention）
    {
        'name': 'Round7_Config12_Combined_Best',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 0,
        'attention_dim': 320,  # 增大Attention
        'num_attention_heads': 1,
        'dropout_rate': 0.45,  # 适中dropout
        'l2_reg': 0.0012,      # 适中L2
        'learning_rate': 0.0001,  # 低学习率
        'epochs': 400,
        'batch_size': 32,
    },
]


def run_optimization():
    """执行参数优化"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # 动态导入主系统
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lstm_model",
        "/Users/Jason/Desktop/code/AI/lstm_attention_carbon_prediction.py"
    )
    lstm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_module)
    
    results = []
    
    print("\n" + "="*80)
    print(" " * 20 + "煤炭价格预测 - 第七轮参数优化")
    print("="*80 + "\n")
    print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数配置数量: {len(PARAM_GRID)}")
    print(f"目标列: coal_price")
    print(f"基于第五轮最优配置 (R²=0.462, 2层LSTM)")
    print(f"第六轮教训: 单层LSTM失败 (R²=0.421)")
    print(f"核心策略: 回归2层LSTM + 微调学习率/正则化/Attention")
    print("\n" + "="*80 + "\n")
    
    for idx, params in enumerate(PARAM_GRID, 1):
        print(f"\n{'='*80}")
        print(f"执行配置 {idx}/{len(PARAM_GRID)}: {params['name']}")
        print(f"{'='*80}\n")
        
        # 更新配置
        lstm_module.CONFIG.update(params)
        lstm_module.CONFIG['target_column'] = 'coal_price'
        
        # 添加默认值
        if 'num_attention_heads' not in lstm_module.CONFIG:
            lstm_module.CONFIG['num_attention_heads'] = 1
        if 'direction_loss_weight' not in lstm_module.CONFIG:
            lstm_module.CONFIG['direction_loss_weight'] = 0.20
        
        print("📋 当前配置:")
        for key, value in params.items():
            if key != 'name':
                print(f"   • {key}: {value}")
        print()
        
        print(f"🔧 使用2层LSTM架构: {params['lstm_units']}-{params['lstm_units_2']}")
        
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
                'best_epoch': len(system.history.history['loss']),
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
    results_df = results_df.sort_values('R2', ascending=False)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/Users/Jason/Desktop/code/AI/parameter/coal_price_optimization_round7_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='完整结果', index=False)
        
        top3 = results_df.head(3)[['config_name', 'R2', 'RMSE', 'MAPE', 
                                    'Direction_Accuracy', 'sequence_length',
                                    'lstm_units', 'lstm_units_2', 'attention_dim',
                                    'dropout_rate', 'learning_rate',
                                    'batch_size', 'epochs']]
        top3.to_excel(writer, sheet_name='Top3配置', index=False)
        
        param_cols = ['config_name', 'sequence_length', 'lstm_units', 'lstm_units_2',
                     'lstm_units_3', 'attention_dim', 'num_attention_heads',
                     'dropout_rate', 'l2_reg', 'learning_rate', 'batch_size', 'epochs']
        params_df = results_df[[col for col in param_cols if col in results_df.columns]]
        params_df.to_excel(writer, sheet_name='参数配置', index=False)
        
        metrics_cols = ['config_name', 'MSE', 'MAE', 'RMSE', 'R2', 'MAPE', 
                       'Direction_Accuracy']
        metrics_df = results_df[metrics_cols]
        metrics_df.to_excel(writer, sheet_name='性能指标', index=False)
    
    print(f"✅ 结果已保存到: {output_path}\n")
    
    print("🏆 Top 3 配置:\n")
    for idx, row in results_df.head(3).iterrows():
        print(f"{row['config_name']}:")
        print(f"   • R² = {row['R2']:.4f}")
        print(f"   • RMSE = {row['RMSE']:.4f}")
        print(f"   • MAPE = {row['MAPE']:.2f}%")
        print(f"   • 方向准确率 = {row['Direction_Accuracy']:.2f}%")
        print()
    
    print("="*80)
    print(f"第七轮优化完成! 最佳R² = {results_df.iloc[0]['R2']:.4f}")
    print(f"相比第五轮最优(R²=0.462): {(results_df.iloc[0]['R2']-0.462)/0.462*100:+.1f}%")
    print(f"相比第六轮最优(R²=0.421): {(results_df.iloc[0]['R2']-0.421)/0.421*100:+.1f}%")
    print(f"相比第二轮最优(R²=0.219): {(results_df.iloc[0]['R2']-0.219)/0.219*100:+.1f}%")
    print("="*80 + "\n")
    
    return results_df


if __name__ == '__main__':
    results = run_optimization()
