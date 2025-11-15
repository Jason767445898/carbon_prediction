import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置全局样式
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (20, 12)

# 按发表时间排序的模型（从最早到最新）
models_timeline = [
    'Mean Baseline',      # 基础统计基准 (最早)
    'Median Baseline',    # 基础统计基准
    # 1967年 - 最近邻算法
    'Decision Tree',      # 1984年 - 分类与回归树
    'LSTM',               # 1997年 - 长短期记忆网络
    'Random Forest',      # 2001年 - 随机森林
    'GradientBoosting',   # 2001年 - 梯度提升
    'XGBoost',            # 2016年 - 极端梯度提升
    'Attention',          # 2017年 - 注意力机制
    'Our Model'          # 当前模型 (最新)
]
# 原始模型数据顺序
models_original = [
    'Mean Baseline', 
    'Median Baseline', 
    'Decision Tree', 
    'Random Forest', 
    'XGBoost', 
    'GradientBoosting', 
    'LSTM', 
    'Attention', 
    'Our Model'
]

# 所有10个模型的指标数据
metrics_data = {
    'R²': [-0.0869, -0.1295, 0.7811, 0.7954, 0.5663, 0.6602, -1.4023, -0.3695, 0.8349],
    'RMSE': [88.5661, 90.2831, 39.7498,  38.4275, 55.9429, 49.5217, 130.0518, 100.4576, 42.1042],
    'MAE': [65.0989, 64.4444, 26.8923,  24.4691, 36.9668, 32.9089, 99.6849, 71.7754, 31.0717],
    'Direction Accuracy': [0.2895, 0.2895, 0.4145, 0.5526, 0.5592, 0.6118, 0.5524, 0.4126, 0.4286],
    'MAPE': [0.0925, 0.0906, 0.0381,  0.0345, 0.0513, 0.0458, 0.1252, 0.0987, 0.0326]
}

# 将MAPE转换为百分比显示
mapes_percentage = [x * 100 for x in metrics_data['MAPE']]

# 创建原始顺序的DataFrame
df_original = pd.DataFrame(metrics_data, index=models_original)
df_original['MAPE_Percentage'] = mapes_percentage

# 按时间线顺序重新排列DataFrame
df_timeline = df_original.reindex(models_timeline)

# 创建图形和子图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 优化主标题排版
fig.suptitle('Carbon Price Prediction Model Performance Comparison\n(Ordered by Publication Timeline)', 
             fontsize=20, fontweight='bold', y=0.98, 
             fontfamily='sans-serif', linespacing=1.5)



# 颜色设置 - 突出显示您的模型为橙色，其他为蓝色
colors_timeline = []
for model in models_timeline:
    if model == 'Our Model':
        colors_timeline.append('#ff7f0e')  # 橙色突出显示您的模型
    else:
        colors_timeline.append('#1f77b4')  # 蓝色用于基线模型

# 柱状图的X轴位置
x_pos = np.arange(len(models_timeline))
bar_width = 0.7

# 1. R² 对比图（时间线顺序）
axes[0, 0].bar(x_pos, df_timeline['R²'], color=colors_timeline, alpha=0.8, width=bar_width)
axes[0, 0].set_title('R² Comparison\n(Higher is Better)', fontweight='bold', pad=12, fontsize=12)
axes[0, 0].set_ylabel('R² Score', fontsize=11)
axes[0, 0].set_ylim(-1.5, 1.0)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

# 2. RMSE 对比图（时间线顺序）
axes[0, 1].bar(x_pos, df_timeline['RMSE'], color=colors_timeline, alpha=0.8, width=bar_width)
axes[0, 1].set_title('RMSE Comparison\n(Lower is Better)', fontweight='bold', pad=12, fontsize=12)
axes[0, 1].set_ylabel('RMSE', fontsize=11)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. MAE 对比图（时间线顺序）
axes[0, 2].bar(x_pos, df_timeline['MAE'], color=colors_timeline, alpha=0.8, width=bar_width)
axes[0, 2].set_title('MAE Comparison\n(Lower is Better)', fontweight='bold', pad=12, fontsize=12)
axes[0, 2].set_ylabel('MAE', fontsize=11)
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[0, 2].grid(axis='y', alpha=0.3)

# 4. 方向准确率对比图（时间线顺序）
axes[1, 0].bar(x_pos, df_timeline['Direction Accuracy'], color=colors_timeline, alpha=0.8, width=bar_width)
axes[1, 0].set_title('Direction Accuracy Comparison\n(Higher is Better)', fontweight='bold', pad=12, fontsize=12)
axes[1, 0].set_ylabel('Direction Accuracy', fontsize=11)
axes[1, 0].set_ylim(0, 0.7)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3)

# 5. MAPE 对比图（时间线顺序）
axes[1, 1].bar(x_pos, df_timeline['MAPE_Percentage'], color=colors_timeline, alpha=0.8, width=bar_width)
axes[1, 1].set_title('MAPE Comparison\n(Lower is Better)', fontweight='bold', pad=12, fontsize=12)
axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[1, 1].grid(axis='y', alpha=0.3)

# 6. 综合性能评分图（时间线顺序）
# 标准化并加权各项指标计算综合得分
r2_normalized = (df_timeline['R²'] - df_timeline['R²'].min()) / (df_timeline['R²'].max() - df_timeline['R²'].min())
rmse_normalized = 1 - ((df_timeline['RMSE'] - df_timeline['RMSE'].min()) / (df_timeline['RMSE'].max() - df_timeline['RMSE'].min()))
mae_normalized = 1 - ((df_timeline['MAE'] - df_timeline['MAE'].min()) / (df_timeline['MAE'].max() - df_timeline['MAE'].min()))
direction_normalized = (df_timeline['Direction Accuracy'] - df_timeline['Direction Accuracy'].min()) / (df_timeline['Direction Accuracy'].max() - df_timeline['Direction Accuracy'].min())
mape_normalized = 1 - ((df_timeline['MAPE_Percentage'] - df_timeline['MAPE_Percentage'].min()) / (df_timeline['MAPE_Percentage'].max() - df_timeline['MAPE_Percentage'].min()))

comprehensive_scores = (r2_normalized * 0.3 + rmse_normalized * 0.2 + mae_normalized * 0.2 + 
                       direction_normalized * 0.2 + mape_normalized * 0.1)

axes[1, 2].bar(x_pos, comprehensive_scores, color=colors_timeline, alpha=0.8, width=bar_width)
axes[1, 2].set_title('Comprehensive Performance Score', fontweight='bold', pad=12, fontsize=12)
axes[1, 2].set_ylabel('Score (0-1)', fontsize=11)
axes[1, 2].set_ylim(0, 1)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(models_timeline, rotation=45, ha='right', fontsize=9)
axes[1, 2].grid(axis='y', alpha=0.3)

# 在柱状图上添加数值标签
for i, ax in enumerate(axes.flat):
    for j, model in enumerate(models_timeline):
        if i == 0:  # R²
            value = df_timeline['R²'].iloc[j]
            if value >= 0:
                ax.text(j, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(j, value - 0.05, f'{value:.3f}', ha='center', va='top', fontsize=8, color='red')
        elif i == 1:  # RMSE
            value = df_timeline['RMSE'].iloc[j]
            ax.text(j, value + 2, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        elif i == 2:  # MAE
            value = df_timeline['MAE'].iloc[j]
            ax.text(j, value + 1.5, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        elif i == 3:  # 方向准确率
            value = df_timeline['Direction Accuracy'].iloc[j]
            ax.text(j, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        elif i == 4:  # MAPE
            value = df_timeline['MAPE_Percentage'].iloc[j]
            ax.text(j, value + 0.5, f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
        elif i == 5:  # 综合得分
            value = comprehensive_scores.iloc[j]
            ax.text(j, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# 优化布局
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.8, label='Historical Models'),
    Patch(facecolor='#ff7f0e', alpha=0.8, label='Our Model')
]
fig.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True, fontsize=12)

# 添加时间线注释
fig.text(0.02, 0.02, 'Timeline: 1960s → 2020s', 
         fontsize=9, alpha=0.7, style='italic')

# 保存和显示图表
plt.savefig('timeline_ordered_model_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# 打印时间线顺序供参考
print("Model Timeline Order (Chronological):")
for i, model in enumerate(models_timeline, 1):
    print(f"{i:2d}. {model}")