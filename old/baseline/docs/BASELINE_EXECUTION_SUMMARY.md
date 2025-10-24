# 碳价格预测 - Baseline五模型运行总结

## 📋 执行概览

已成功创建并启动了包含五种传统序列模型的Baseline实验脚本，用于碳价格预测任务。

**启动时间**: 2025-10-24 20:37:08  
**脚本位置**: `/Users/Jason/Desktop/code/AI/run_baseline_models.py`  
**日志位置**: `/Users/Jason/Desktop/code/AI/baseline_output.log`  
**输出目录**: `/Users/Jason/Desktop/code/AI/outputs/baseline/`

---

## 🎯 五大Baseline模型

### 1. **RNN (循环神经网络)**
- **层次结构**: SimpleRNN(64) → SimpleRNN(32) → Dense(16) → Output
- **参数**: dropout=0.2, 学习率=0.001
- **特点**: 简单的循环结构，捕捉序列依赖关系

### 2. **GRU (门控循环单元)**
- **层次结构**: GRU(64) → GRU(32) → Dense(16) → Output
- **参数**: dropout=0.2, 学习率=0.001
- **特点**: 相比LSTM更轻量级，计算效率高

### 3. **LSTM (长短期记忆网络)**
- **层次结构**: LSTM(64) → LSTM(32) → Dense(16) → Output
- **参数**: dropout=0.2, 学习率=0.001
- **特点**: 双路记忆机制，处理长期依赖效果好

### 4. **Transformer (注意力机制模型)**
- **层次结构**: 
  - Dense投影(64维)
  - MultiHeadAttention(4头)
  - FeedForward网络
  - GlobalAveragePooling → Dense(32) → Output
- **参数**: key_dim=16, dropout=0.2, 学习率=0.001
- **特点**: 自注意力机制，并行计算能力强

### 5. **AutoFormer (自动回归Transformer)**
- **层次结构**:
  - Conv1D(32) → MaxPooling → UpSampling (提取局部特征)
  - MultiHeadAttention(4头) (全局依赖)
  - FeedForward网络
  - GlobalAveragePooling → Dense(32) → Output
- **参数**: key_dim=8, dropout=0.2, 学习率=0.001
- **特点**: 结合卷积和注意力，适合时间序列

---

## 📊 数据信息

| 指标 | 数值 |
|-----|------|
| **原始数据** | 1,247 条记录 × 21 列 |
| **预处理后** | 1,237 条记录 × 28 列 |
| **特征数** | 27 个（排除目标列coal_price） |
| **序列长度** | 60 时间步 |
| **训练集** | 865 条 (70%) |
| **验证集** | 124 条 (10%) |
| **测试集** | 248 条 (20%) |

### 特征列表

**原始特征** (19列):
- coal_price, oil_price, gas_price, carbon_price_hb_ea
- transactionamount_hb_ea, aqi_hb, highest_temperature
- log_coal_price, log_oil_price, log_gas_price, ...
- log_coal_price_sqr, log_oil_price_sqr, ...

**工程特征** (9列):
- 移动平均: ma_5, ma_10, ma_20, ma_30
- 滞后特征: coal_price_lag_1, coal_price_lag_2, coal_price_lag_3, coal_price_lag_5, coal_price_lag_10

**删除的列**:
- var9, var10 (完全为NaN，无预测价值)

---

## ⚙️ 模型配置

```python
CONFIG = {
    'target_column': 'coal_price',      # 目标列：煤炭价格
    'sequence_length': 60,              # 序列长度：60天
    'test_size': 0.2,                   # 测试集比例：20%
    'val_size': 0.1,                    # 验证集比例：10%
    'epochs': 150,                      # 训练轮数：150
    'batch_size': 32,                   # 批量大小：32
    'random_state': 42                  # 随机种子：42
}
```

### 训练参数

| 参数 | 值 |
|-----|-----|
| **损失函数** | MSE (Mean Squared Error) |
| **优化器** | Adam (学习率=0.001) |
| **早停** | 监控val_loss, patience=15 |
| **学习率衰减** | factor=0.5, patience=10, min_lr=1e-6 |
| **评估指标** | MAE (平均绝对误差) |

---

## 📈 预期输出

### 生成的文件

1. **CSV结果文件**
   - `baseline_YYYYMMDD_HHMMSS_baseline_results.csv`
   - 包含5个模型的性能对比（R², RMSE, MAE, MAPE, 方向准确率）

2. **可视化图表**
   - `baseline_YYYYMMDD_HHMMSS_baseline_comparison.png` - 模型性能对比柱状图
   - `baseline_YYYYMMDD_HHMMSS_baseline_predictions.png` - 预测结果对比（最后200个点）

3. **最佳模型权重**
   - `baseline_YYYYMMDD_HHMMSS_rnn_best.h5`
   - `baseline_YYYYMMDD_HHMMSS_gru_best.h5`
   - `baseline_YYYYMMDD_HHMMSS_lstm_best.h5`
   - `baseline_YYYYMMDD_HHMMSS_transformer_best.h5`
   - `baseline_YYYYMMDD_HHMMSS_autoformer_best.h5`

### 评估指标说明

| 指标 | 说明 | 理想值 |
|-----|------|-------|
| **R²** | 决定系数，越接近1越好 | > 0.8 (优秀) |
| **RMSE** | 均方根误差，越小越好 | 越小越好 |
| **MAE** | 平均绝对误差，越小越好 | 越小越好 |
| **MAPE** | 平均百分比误差，越小越好 | < 5% (优秀) |
| **方向准确率** | 预测价格涨跌方向准确率 | > 55% (较好) |

---

## 🔄 运行流程

```
加载数据 (data.dta)
    ↓
数据预处理 (处理NaN、特征工程)
    ↓
数据分割 (70% train, 10% val, 20% test)
    ↓
准备序列数据 (60步长窗口)
    ↓
数据标准化 (MinMaxScaler [0,1])
    ↓
────────────────────────────────────
│ RNN训练 → GRU训练 → LSTM训练 →   │
│ Transformer训练 → AutoFormer训练  │
────────────────────────────────────
    ↓
评估和对比 (在测试集上评估)
    ↓
可视化结果 (生成对比图表)
    ↓
输出完整报告
```

---

## ⏱️ 预期执行时间

| 模型 | 估计时间 | 说明 |
|-----|---------|------|
| RNN | 3-5分钟 | 轻量级，最快 |
| GRU | 4-6分钟 | 轻量级，与RNN相近 |
| LSTM | 5-8分钟 | 中等复杂度 |
| Transformer | 6-10分钟 | 注意力机制计算量大 |
| AutoFormer | 7-12分钟 | 最复杂，但通常不会训满150轮 |
| **总计** | **25-40分钟** | 取决于早停和系统性能 |

---

## 📊 监控进度

### 查看实时日志
```bash
tail -f /Users/Jason/Desktop/code/AI/baseline_output.log
```

### 运行状态检查
```bash
ps aux | grep run_baseline_models.py
```

### 后台进程 PID
进程ID: 36113 (启动于 2025-10-24 20:37:08)

---

## 💡 预期结果对标

基于项目历史的Baseline报告数据：

| 模型 | 历史R² | 历史RMSE |
|-----|--------|---------|
| 随机森林 | 0.9430 | 40.69 |
| 梯度提升 | 0.8756 | 60.13 |
| XGBoost | 0.6603 | 99.38 |
| LSTM | -0.2348 | 115.15 |
| Transformer | -2.0934 | 182.25 |

> **注**: 这次的五模型实验可能会有不同的结果，因为：
> - 数据预处理方式改进
> - 模型架构优化
> - 参数调整
> - 防止过拟合的正则化策略

---

## ✅ 完成检查清单

- [x] 创建Baseline脚本 (`run_baseline_models.py`)
- [x] 实现数据加载和预处理
- [x] 构建5种序列模型（RNN, GRU, LSTM, Transformer, AutoFormer）
- [x] 配置训练参数和回调函数
- [x] 后台启动防休眠训练
- [x] 创建日志输出
- [x] 预留可视化功能

---

## 🎯 后续步骤

### 1. **等待训练完成**
   - 监控 `baseline_output.log` 查看进度
   - 预计 25-40 分钟完成全部五个模型训练

### 2. **查看结果**
   ```bash
   ls -lah /Users/Jason/Desktop/code/AI/outputs/baseline/
   ```

### 3. **分析对比**
   - 查看 `baseline_YYYYMMDD_HHMMSS_baseline_results.csv`
   - 查看对比图表 `baseline_comparison.png`

### 4. **进一步优化** (可选)
   - 根据最佳模型进行超参数微调
   - 尝试集成学习（多模型投票）
   - 添加额外特征
   - 调整序列长度

---

## 📞 技术支持

### 常见问题

**Q: 训练是否仍在进行?**  
A: 查看 `ps aux | grep caffeinate` 确认进程是否运行

**Q: 如何停止训练?**  
A: `kill 36113` (进程ID) 或 `pkill -f caffeinate`

**Q: 如何查看完整输出?**  
A: `cat /Users/Jason/Desktop/code/AI/baseline_output.log`

**Q: 输出文件在哪里?**  
A: `/Users/Jason/Desktop/code/AI/outputs/baseline/`

---

**报告生成时间**: 2025-10-24 20:37:08  
**脚本版本**: v1.0  
**状态**: 🟢 运行中

---
