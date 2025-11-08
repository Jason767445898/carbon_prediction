#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 data.dta 中指定列的数据
区分训练集和验证集（测试集）
支持通过命令行参数选择要可视化的列
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数（与 simple_lstm_attention_coal.py 保持一致）
CONFIG = {
    'data_file': 'data.dta',
    'target_column': 'coal_price',  # 默认列，可通过命令行参数覆盖
    'test_size': 0.2,           # 测试集比例：20%
    'validation_size': 0.1,     # 验证集比例：10%
    'sequence_length': 30,      # 序列长度（用于计算有效数据点）
}

# 可用的数据列及其中文名称
COLUMN_NAMES = {
    'coal_price': '煤炭价格',
    'oil_price': '石油价格',
    'gas_price': '天然气价格',
    'carbon_price_hb_ea': '碳价格(湖北)',
    'transactionamount_hb_ea': '交易量(湖北)',
    'aqi_hb': '空气质量指数',
    'highest_temperature': '最高温度',
    'log_coal_price': '对数煤炭价格',
    'log_oil_price': '对数石油价格',
    'log_gas_price': '对数天然气价格',
    'log_carbon_price_hb_ea': '对数碳价格',
    'log_transactionamount_hb_ea': '对数交易量',
    'log_aqi_hb': '对数空气质量指数',
    'log_highest_temperature': '对数最高温度',
    'log_coal_price_sqr': '对数煤炭价格平方',
    'log_oil_price_sqr': '对数石油价格平方',
    'log_gas_price_sqr': '对数天然气价格平方',
    'log_transactionamount_hb_ea_sqr': '对数交易量平方',
    'log_aqi_hb_sqr': '对数空气质量指数平方',
}

OUTPUT_DIR = 'outputs/data_info'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='可视化 data.dta 中指定列的时序数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
可用的列名:
  coal_price                    煤炭价格
  oil_price                     石油价格
  gas_price                     天然气价格
  carbon_price_hb_ea            碳价格(湖北)
  transactionamount_hb_ea       交易量(湖北)
  aqi_hb                        空气质量指数
  highest_temperature           最高温度
  log_coal_price                对数煤炭价格
  log_oil_price                 对数石油价格
  log_gas_price                 对数天然气价格
  log_carbon_price_hb_ea        对数碳价格
  log_transactionamount_hb_ea   对数交易量
  log_aqi_hb                    对数空气质量指数
  log_highest_temperature       对数最高温度
  log_coal_price_sqr            对数煤炭价格平方
  log_oil_price_sqr             对数石油价格平方
  log_gas_price_sqr             对数天然气价格平方
  log_transactionamount_hb_ea_sqr  对数交易量平方
  log_aqi_hb_sqr                对数空气质量指数平方

示例:
  python visualize_coal_price.py                      # 默认显示 coal_price
  python visualize_coal_price.py -c oil_price         # 显示石油价格
  python visualize_coal_price.py --column carbon_price_hb_ea  # 显示碳价格
  python visualize_coal_price.py --list               # 列出所有可用列
        ''')
    
    parser.add_argument(
        '-c', '--column',
        type=str,
        default='coal_price',
        help='要可视化的列名 (默认: coal_price)'
    )
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='列出所有可用的列名'
    )
    
    return parser.parse_args()

def list_available_columns():
    """列出所有可用的列"""
    print("=" * 80)
    print(" " * 25 + "可用的数据列")
    print("=" * 80 + "\n")
    
    print(f"{'列名':<40} {'中文名称'}")
    print("-" * 80)
    
    for col, name in COLUMN_NAMES.items():
        print(f"{col:<40} {name}")
    
    print("\n" + "=" * 80)
    print(f"总计: {len(COLUMN_NAMES)} 个可用列")
    print("=" * 80 + "\n")

def load_and_visualize(target_column='coal_price'):
    """加载数据并可视化
    
    Args:
        target_column: 要可视化的列名
    """
    # 获取中文列名
    column_display_name = COLUMN_NAMES.get(target_column, target_column)
    
    print("=" * 80)
    print(" " * 20 + f"{column_display_name} 数据可视化")
    print("=" * 80 + "\n")
    
    # 1. 加载数据
    print(f"📊 加载数据文件: {CONFIG['data_file']}")
    
    if not os.path.exists(CONFIG['data_file']):
        raise FileNotFoundError(f"文件不存在: {CONFIG['data_file']}")
    
    data = pd.read_stata(CONFIG['data_file'])
    
    # 转换日期列
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    # 🔥 筛选2017-2021年的数据
    original_shape = data.shape
    data = data[(data.index.year >= 2017) & (data.index.year <= 2024)]
    
    print(f"✅ 数据加载成功")
    print(f"   • 原始数据形状: {original_shape}")
    print(f"   • 筛选后数据形状: {data.shape}")
    print(f"   • 时间范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"   • 筛选条件: 2017-2021年")
    print(f"   • 列名: {list(data.columns)}\n")
    
    # 检查目标列
    if target_column not in data.columns:
        print(f"\n❌ 错误: 列 '{target_column}' 不存在于数据中")
        print(f"\n可用的列: {list(data.columns)}\n")
        print("使用 --list 参数查看所有可用列名\n")
        sys.exit(1)
    
    # 2. 提取目标列数据
    target_data = data[target_column].copy()
    
    # 移除缺失值
    target_data = target_data.dropna()
    
    print(f"📈 {column_display_name} ({target_column}) 数据统计:")
    print(f"   • 有效数据点: {len(target_data)}")
    print(f"   • 最小值: {target_data.min():.2f}")
    print(f"   • 最大值: {target_data.max():.2f}")
    print(f"   • 平均值: {target_data.mean():.2f}")
    print(f"   • 标准差: {target_data.std():.2f}\n")
    
    # 3. 划分训练集、验证集、测试集
    # 考虑序列长度的影响
    n = len(target_data)
    effective_n = n - CONFIG['sequence_length']  # 创建序列后的有效样本数
    
    # 计算分割点（基于有效样本数）
    train_end_idx = int(effective_n * (1 - CONFIG['test_size'] - CONFIG['validation_size'])) + CONFIG['sequence_length']
    val_end_idx = int(effective_n * (1 - CONFIG['test_size'])) + CONFIG['sequence_length']
    
    # 分割数据
    train_data = target_data.iloc[:train_end_idx]
    val_data = target_data.iloc[train_end_idx:val_end_idx]
    test_data = target_data.iloc[val_end_idx:]
    
    print(f"📊 数据集划分:")
    print(f"   • 训练集: {len(train_data)} 个数据点 ({len(train_data)/len(target_data)*100:.1f}%)")
    print(f"   • 验证集: {len(val_data)} 个数据点 ({len(val_data)/len(target_data)*100:.1f}%)")
    print(f"   • 测试集: {len(test_data)} 个数据点 ({len(test_data)/len(target_data)*100:.1f}%)")
    print(f"   • 训练集时间范围: {train_data.index[0]} 到 {train_data.index[-1]}")
    print(f"   • 验证集时间范围: {val_data.index[0]} 到 {val_data.index[-1]}")
    print(f"   • 测试集时间范围: {test_data.index[0]} 到 {test_data.index[-1]}\n")
    
    # 4. 绘制折线图
    print("🎨 生成可视化图表...")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 绘制训练集
    ax.plot(train_data.index, train_data.values, 
            label='训练集 (Training Set)', 
            color='#2E86C1', linewidth=1.5, alpha=0.9)
    
    # 绘制验证集
    ax.plot(val_data.index, val_data.values, 
            label='验证集 (Validation Set)', 
            color='#28B463', linewidth=1.5, alpha=0.9)
    
    # 绘制测试集
    ax.plot(test_data.index, test_data.values, 
            label='测试集 (Test Set)', 
            color='#E74C3C', linewidth=1.5, alpha=0.9)
    
    # 添加分割线
    ax.axvline(x=train_data.index[-1], color='gray', linestyle='--', 
               linewidth=1, alpha=0.7, label='训练/验证分割点')
    ax.axvline(x=val_data.index[-1], color='gray', linestyle='--', 
               linewidth=1, alpha=0.7, label='验证/测试分割点')
    
    # 设置标题和标签
    ax.set_title(f'{column_display_name} ({target_column}) 时序数据 - 训练集/验证集/测试集划分', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('日期 (Date)', fontsize=12)
    ax.set_ylabel(f'{column_display_name} ({target_column})', fontsize=12)
    
    # 图例
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息文本框
    stats_text = f'训练集: {len(train_data)} 点\n验证集: {len(val_data)} 点\n测试集: {len(test_data)} 点\n总计: {len(target_data)} 点'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 旋转日期标签
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图表
    output_filename = f'{target_column}_visualization.png'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果指定了 --list，列出所有可用列并退出
    if args.list:
        list_available_columns()
        sys.exit(0)
    
    # 如果指定了特定列，只生成该列的图表
    if args.column != 'coal_price' or len(sys.argv) > 1:
        load_and_visualize(target_column=args.column)
    else:
        # 默认情况下，依次生成所有列的折线图
        print("\n" + "=" * 80)
        print("" * 25 + "批量生成所有列的可视化图表")
        print("=" * 80 + "\n")
        
        total_columns = len(COLUMN_NAMES)
        for idx, (col, name) in enumerate(COLUMN_NAMES.items(), 1):
            print(f"\n{'='*80}")
            print(f"进度: [{idx}/{total_columns}] 正在处理: {name} ({col})")
            print(f"{'='*80}\n")
            
            try:
                load_and_visualize(target_column=col)
            except Exception as e:
                print(f"\n⚠️ 生成 {name} ({col}) 图表时出错: {e}\n")
                continue
        
        print("\n" + "=" * 80)
        print("" * 20 + f"✅ 已完成所有 {total_columns} 个列的可视化")
        print(f" 图表保存位置: {OUTPUT_DIR}")
        print("=" * 80 + "\n")
