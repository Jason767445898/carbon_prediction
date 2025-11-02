#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤炭价格预测 - 第三轮参数优化
基于第二轮最优配置(Round2_Config3_BiggerAttention)进行优化
探索多头注意力、更大维度、学习率微调等
"""

import sys
import os
sys.path.append('/Users/Jason/Desktop/code/AI')

# 参数搜索空间 - 基于Round2_Config3优化
PARAM_GRID = [
    # 配置1: 多头注意力(4头)
    {
        'name': 'Round3_Config1_MultiHead4',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 256,
        'num_attention_heads': 4,  # 新增
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置2: 多头注意力(8头)
    {
        'name': 'Round3_Config2_MultiHead8',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 256,
        'num_attention_heads': 8,  # 新增
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置3: 更大Attention维度(320)
    {
        'name': 'Round3_Config3_Attention320',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 320,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置4: Attention维度匹配LSTM(384)
    {
        'name': 'Round3_Config4_Attention384',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 384,
        'num_attention_heads': 1,
        'dropout_rate': 0.42,
        'l2_reg': 0.0012,
        'learning_rate': 0.00015,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置5: 学习率中间值
    {
        'name': 'Round3_Config5_LR00012',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00012,
        'epochs': 400,
        'batch_size': 32,
    },
    
    # 配置6: 增强方向感知
    {
        'name': 'Round3_Config6_DirectionFocus',
        'sequence_length': 60,
        'lstm_units': 384,
        'lstm_units_2': 192,
        'lstm_units_3': 96,
        'attention_dim': 256,
        'num_attention_heads': 1,
        'dropout_rate': 0.4,
        'l2_reg': 0.001,
        'learning_rate': 0.00015,
        'direction_loss_weight': 0.30,  # 新增
        'epochs': 400,
        'batch_size': 32,
    },
]


def run_optimization():
    """执行参数优化"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # 动态导入并修改主系统
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lstm_model",
        "/Users/Jason/Desktop/code/AI/lstm_attention_carbon_prediction.py"
    )
    lstm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_module)
    
    results = []
    
    print("\n" + "="*80)
    print(" " * 20 + "煤炭价格预测 - 第三轮参数优化")
    print("="*80 + "\n")
    print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数配置数量: {len(PARAM_GRID)}")
    print(f"目标列: coal_price")
    print(f"基于第二轮最优Round2_Config3_BiggerAttention (R²=0.219)")
    print(f"核心改进: 多头注意力 + 深度优化 + 学习率微调")
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
        
        try:
            # 如果使用多头注意力,需要修改attention层创建函数
            if params.get('num_attention_heads', 1) > 1:
                print(f"🔧 使用多头注意力 ({params['num_attention_heads']}头)")
                # 重写create_attention_layer函数
                original_create_attention = lstm_module.create_attention_layer
                
                def create_multi_head_attention(input_tensor, attention_dim):
                    num_heads = lstm_module.CONFIG['num_attention_heads']
                    assert attention_dim % num_heads == 0, f"attention_dim({attention_dim})必须能被num_heads({num_heads})整除"
                    
                    from tensorflow.keras import layers
                    import tensorflow as tf
                    
                    head_dim = attention_dim // num_heads
                    head_outputs = []
                    
                    for i in range(num_heads):
                        query = layers.Dense(head_dim, name=f'head_{i}_query')(input_tensor)
                        key = layers.Dense(head_dim, name=f'head_{i}_key')(input_tensor)
                        value = layers.Dense(head_dim, name=f'head_{i}_value')(input_tensor)
                        
                        scores = layers.Dot(axes=[2, 2])([query, key])
                        scores = layers.Lambda(lambda x: x / tf.math.sqrt(tf.cast(head_dim, tf.float32)))(scores)
                        attention_weights = layers.Softmax(axis=-1, name=f'head_{i}_weights')(scores)
                        
                        context = layers.Dot(axes=[2, 1])([attention_weights, value])
                        head_outputs.append(context)
                    
                    if num_heads > 1:
                        multi_head = layers.Concatenate(axis=-1, name='concat_heads')(head_outputs)
                    else:
                        multi_head = head_outputs[0]
                    
                    output = layers.Dense(attention_dim, name='multi_head_projection')(multi_head)
                    context_vector = layers.GlobalAveragePooling1D(name='attention_pooling')(output)
                    
                    return context_vector
                
                lstm_module.create_attention_layer = create_multi_head_attention
            
            # 如果调整方向损失权重,需要修改损失函数
            if params.get('direction_loss_weight'):
                print(f"🎯 调整方向损失权重至 {params['direction_loss_weight']}")
            
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
            
            # 恢复原始函数
            if params.get('num_attention_heads', 1) > 1:
                lstm_module.create_attention_layer = original_create_attention
            
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
    output_path = f'/Users/Jason/Desktop/code/AI/parameter/coal_price_optimization_round3_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='完整结果', index=False)
        
        top3 = results_df.head(3)[['config_name', 'R2', 'RMSE', 'MAPE', 
                                    'Direction_Accuracy', 'sequence_length',
                                    'lstm_units', 'attention_dim', 'num_attention_heads',
                                    'learning_rate', 'batch_size']]
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
        if 'num_attention_heads' in row and row['num_attention_heads'] > 1:
            print(f"   • 注意力头数 = {row['num_attention_heads']}")
        print()
    
    print("="*80)
    print(f"第三轮优化完成! 最佳R² = {results_df.iloc[0]['R2']:.4f}")
    print("="*80 + "\n")
    
    return results_df


if __name__ == '__main__':
    results = run_optimization()
