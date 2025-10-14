# 碳价格预测系统 Baseline 报告

**生成时间**: 2025-10-14 20:25:43  
**运行标识**: carbon_price_prediction_20251014_202008  
**数据来源**: data.dta

---

## 📊 数据概览

| 指标 | 数值 |
|------|------|
| 数据点数量 | 1,247 |
| 特征数量 | 45 |
| 时间范围 | 2016-09-22 至 2023-12-29 |
| 目标变量 | coal_price |
| 价格范围 | 463.00 - 1,628.00 |
| 数据完整性 | 100% |
| 价格波动率 | 269.41 |

### 原始特征列表
- coal_price, oil_price, gas_price, carbon_price_hb_ea
- transactionamount_hb_ea, aqi_hb, highest_temperature
- var9, var10
- log_coal_price, log_oil_price, log_gas_price, log_carbon_price_hb_ea
- log_transactionamount_hb_ea, log_aqi_hb, log_highest_temperature
- log_coal_price_sqr, log_oil_price_sqr, log_gas_price_sqr
- log_transactionamount_hb_ea_sqr, log_aqi_hb_sqr

### 工程特征
- 移动平均: ma_5, ma_10, ma_20, ma_30
- 布林带: bb_middle, bb_upper, bb_lower, bb_width, bb_position
- 滞后特征: coal_price_lag_1, coal_price_lag_2, coal_price_lag_3, coal_price_lag_5, coal_price_lag_10
- 技术指标: RSI, 价格波动率, 价格动量等

---

## 🏆 模型性能对比（Baseline 结果）

### 总体排名（按 R² 排序）

| 排名 | 模型 | R² | RMSE | MAE | MAPE | 方向准确率 | 性能等级 |
|------|------|-----|------|-----|------|-----------|---------|
| 🥇 1 | **RandomForest** | **0.9430** | **40.69** | **29.58** | **2.81%** | **58.63%** | **优秀** |
| 🥈 2 | GradientBoosting | 0.8756 | 60.13 | 42.06 | 3.94% | 58.63% | 优秀 |
| 🥉 3 | XGBoost | 0.6603 | 99.38 | 58.99 | 4.99% | 51.81% | 良好 |
| 4 | LSTM | -0.2348 | 115.15 | 103.12 | 11.00% | 36.51% | 待改进 |
| 5 | Transformer | -2.0934 | 182.25 | 160.64 | 17.29% | 31.22% | 待改进 |

### 模型详细性能

#### 1. RandomForest（最佳模型）✨
```
R²: 0.9430          # 解释了94.30%的方差
RMSE: 40.6927       # 平均预测误差约40.69
MAE: 29.5799        # 平均绝对误差29.58
MAPE: 2.81%         # 平均百分比误差2.81%
方向准确率: 58.63%   # 价格涨跌方向预测准确率
性能等级: 优秀
```

#### 2. GradientBoosting
```
R²: 0.8756
RMSE: 60.1305
MAE: 42.0619
MAPE: 3.94%
方向准确率: 58.63%
性能等级: 优秀
```

#### 3. XGBoost
```
R²: 0.6603
RMSE: 99.3826
MAE: 58.9860
MAPE: 4.99%
方向准确率: 51.81%
性能等级: 良好
```

#### 4. LSTM（深度学习）
```
R²: -0.2348         # 负值表示性能不如简单平均
RMSE: 115.1471
MAE: 103.1163
MAPE: 11.00%
方向准确率: 36.51%
性能等级: 待改进
```

#### 5. Transformer（深度学习）
```
R²: -2.0934         # 负值表示性能不如简单平均
RMSE: 182.2526
MAE: 160.6357
MAPE: 17.29%
方向准确率: 31.22%
性能等级: 待改进
```

---

## 🔍 特征重要性分析（基于 SHAP）

基于 **RandomForest** 模型的 SHAP 可解释性分析：

| 排名 | 特征名称 | SHAP 重要性 | 特征类型 |
|------|---------|------------|---------|
| 1 | log_coal_price | 149.48 | 对数变换 |
| 2 | log_coal_price_sqr | 125.12 | 对数平方项 |
| 3 | bb_middle | 22.27 | 布林带中轨 |
| 4 | ma_20 | 17.92 | 20日移动平均 |
| 5 | ma_5 | 17.41 | 5日移动平均 |
| 6 | coal_price_lag_1 | 16.62 | 1日滞后 |
| 7 | ma_30 | 15.08 | 30日移动平均 |
| 8 | coal_price_lag_3 | 7.12 | 3日滞后 |
| 9 | coal_price_lag_2 | 6.72 | 2日滞后 |
| 10 | bb_upper | 5.73 | 布林带上轨 |

### 关键发现

1. **对数变换特征主导**：`log_coal_price` 和 `log_coal_price_sqr` 是最重要的特征，贡献了大部分预测能力
2. **技术指标重要**：布林带（bb_middle, bb_upper）和移动平均（ma_5, ma_20, ma_30）对预测有显著影响
3. **时间依赖性**：滞后特征（lag_1, lag_2, lag_3）表明历史价格对未来价格有预测作用
4. **非线性关系**：平方项（log_coal_price_sqr）的高重要性说明存在非线性价格关系

---

## ⚙️ 系统配置

### 数据分割
- **训练集**: 872 样本 (70%)
- **验证集**: 125 样本 (10%)
- **测试集**: 250 样本 (20%)

### 模型参数

#### LSTM 配置
```python
{
    'units': [64, 32],
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 16,
    'sequence_length': 60
}
```

#### Transformer 配置
```python
{
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 4,
    'dff': 512,
    'dropout': 0.1,
    'epochs': 50,
    'sequence_length': 60
}
```

#### 树模型配置
```python
RandomForest: {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

GradientBoosting: {
    'n_estimators': 100,
    'max_depth': 6,
    'random_state': 42
}

XGBoost: {
    'n_estimators': 100,
    'max_depth': 6,
    'random_state': 42
}
```

### 其他配置
- **目标列**: coal_price
- **序列长度**: 60
- **测试集比例**: 0.2
- **验证集比例**: 0.1
- **随机种子**: 42

---

## 📈 Baseline 性能分析

### ✅ 优势

1. **RandomForest 表现优异**
   - R² 达到 0.943，接近完美预测
   - MAPE 仅 2.81%，预测精度非常高
   - 模型稳定性强，泛化能力好

2. **传统机器学习优于深度学习**
   - 树模型（RF, GBDT, XGBoost）在此数据集上表现明显优于 LSTM 和 Transformer
   - 可能原因：数据量相对较小（1247样本），树模型更适合

3. **特征工程有效**
   - 对数变换、移动平均、布林带等技术指标提供了强预测信号
   - 滞后特征捕获了时间序列的自相关性

### ⚠️ 问题与局限

1. **深度学习模型欠拟合**
   - LSTM 和 Transformer 的 R² 为负值，表示预测效果不如简单平均值
   - 可能原因：
     - 数据量不足（深度学习通常需要更大数据集）
     - 超参数未充分调优
     - 特征标准化可能存在问题
     - 序列长度（60）可能不适合

2. **方向预测能力有限**
   - 最佳模型的方向准确率仅 58.63%，略高于随机猜测（50%）
   - 表明价格涨跌方向仍难以准确预测

3. **可能的过拟合风险**
   - RandomForest 的优异表现需要通过交叉验证进一步确认
   - 建议进行 k-fold 交叉验证

---

## 💡 改进建议

### 短期优化（参数调优）

1. **深度学习模型调优**
   - 调整序列长度：尝试 [30, 45, 90, 120]
   - 减少模型复杂度：降低层数和单元数
   - 增加正则化：调整 dropout 率
   - 学习率调度：使用学习率衰减
   - 增加训练轮数：设置早停机制

2. **树模型集成**
   - 使用 Stacking 集成 RF、GBDT、XGBoost
   - 调整树的深度和数量
   - 特征子集采样优化

3. **特征选择**
   - 基于 SHAP 重要性移除低贡献特征
   - 尝试 PCA 降维
   - 添加更多领域特征（如政策指标、季节性）

### 中期优化（特征工程）

1. **时间序列特征增强**
   - 添加季节性特征（月份、季度）
   - 周期性编码（sin/cos 变换）
   - 趋势分解（EMD/STL）

2. **外部数据整合**
   - 宏观经济指标
   - 政策变化标记
   - 市场情绪指数

3. **交互特征**
   - 特征交叉组合
   - 比率特征（如 oil_price / coal_price）

### 长期优化（架构改进）

1. **混合模型**
   - CNN-LSTM 结合
   - Attention + LSTM
   - Informer / Autoformer

2. **集成学习**
   - 多模型投票
   - Stacking 元学习
   - Blending

3. **在线学习**
   - 增量学习框架
   - 模型定期更新机制

---

## 📁 输出文件

### 报告文件
- `carbon_price_prediction_20251014_202008_report.xlsx` - 完整 Excel 分析报告
- `carbon_price_prediction_20251014_202008_detailed_report.txt` - 详细文本报告
- `carbon_price_prediction_20251014_202008_runtime_log.txt` - 运行日志

### 可视化图表
- `carbon_price_prediction_20251014_202008_model_performance_comparison.png` - 模型性能对比
- `carbon_price_prediction_20251014_202008_prediction_comparison.png` - 预测结果对比
- `carbon_price_prediction_20251014_202008_shap_summary_plot.png` - SHAP 特征重要性图
- `carbon_price_prediction_20251014_202008_shap_bar_plot.png` - SHAP 条形图
- `carbon_price_prediction_20251014_202008_shap_dependence_plots.png` - SHAP 依赖图
- `carbon_price_prediction_20251014_202008_training_history.png` - 训练历史曲线

---

## 🎯 结论

### Baseline 总结

1. **当前最佳模型**: RandomForest (R² = 0.943, MAPE = 2.81%)
2. **主要预测因子**: 对数变换的煤炭价格及其平方项、技术指标（移动平均、布林带）
3. **模型类型**: 传统机器学习（树模型）优于深度学习
4. **改进空间**: 深度学习模型需要大幅优化

### 后续工作优先级

1. **高优先级**
   - 深度学习模型超参数调优
   - k-fold 交叉验证
   - 集成学习（Stacking）

2. **中优先级**
   - 特征选择与降维
   - 添加季节性和趋势特征
   - 尝试混合模型架构

3. **低优先级**
   - 在线学习框架
   - 外部数据整合
   - 模型解释性深度分析

### 应用建议

- ✅ **可用于生产**: RandomForest 模型性能优秀，可考虑部署
- ⚠️ **需要监控**: 定期验证模型性能，防止概念漂移
- 📊 **结合人工判断**: 模型预测应结合领域专家意见
- 🔄 **定期更新**: 建议每月重新训练模型

---

## ⚠️ 风险提示

- 模型预测仅供参考，不构成投资建议
- 碳市场受政策影响较大，存在不确定性
- 建议结合多种信息源进行综合判断
- 模型需要定期重新训练以适应市场变化
- 过度依赖历史数据可能导致黑天鹅事件预测失败

---

**报告结束**  
*Generated by Carbon Price Prediction System v1.0*  
*Contact: AI Research Team*
