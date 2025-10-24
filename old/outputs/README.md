# 📊 输出结果目录

> 本目录用于存放系统运行生成的所有输出文件  
> **自动生成** | **不需要手动创建**

---

## 📁 目录结构

```
outputs/
├── logs/                   # 📝 运行日志
│   ├── carbon_prediction_YYYYMMDD_HHMMSS_detailed_report.txt
│   └── carbon_prediction_YYYYMMDD_HHMMSS_runtime_log.txt
│
├── reports/                # 📊 Excel 报告
│   └── carbon_prediction_YYYYMMDD_HHMMSS_report.xlsx
│
└── visualizations/         # 📈 可视化图表
    ├── *_model_performance_comparison.png
    ├── *_prediction_comparison.png
    ├── *_shap_summary_plot.png
    ├── *_shap_bar_plot.png
    ├── *_shap_dependence_plots.png
    └── *_training_history.png
```

---

## 📝 logs/ - 运行日志

### 文件类型

#### 1. detailed_report.txt - 详细分析报告

**内容包括**：
- 📊 数据概览统计
- 🏆 模型性能对比
- 🔍 特征重要性排序
- 📈 预测结果摘要
- ⚙️ 系统配置信息
- ✅ 数据质量报告

**示例**：
```
carbon_prediction_20251017_143022_detailed_report.txt
```

#### 2. runtime_log.txt - 运行过程日志

**内容包括**：
- 🕐 运行开始时间
- 📥 数据加载信息
- 🔧 预处理步骤
- 🤖 模型训练进度
- 📊 评估结果
- ✅ 运行完成时间

**示例**：
```
carbon_prediction_20251017_143022_runtime_log.txt
```

---

## 📊 reports/ - Excel 报告

### 文件结构

**文件名格式**：
```
carbon_prediction_YYYYMMDD_HHMMSS_report.xlsx
```

### Excel 工作表说明

| 工作表名称 | 内容说明 |
|----------|---------|
| **数据概要** | 数据集的描述性统计信息 |
| **模型性能** | 所有模型的性能指标对比 |
| **特征重要性** | SHAP分析的特征重要性排序 |
| **预测结果** | 实际值、预测值和误差详情 |
| **系统配置** | 所有配置参数和系统信息 |
| **数据质量** | 数据完整性、质量指标 |

### 性能指标

报告中包含以下评估指标：
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数（最重要）
- **MAPE**: 平均绝对百分比误差
- **方向准确率**: 趋势预测准确性

---

## 📈 visualizations/ - 可视化图表

### 图表类型

#### 1. model_performance_comparison.png
**内容**：所有模型的性能指标对比柱状图
- 对比5个核心指标：MSE, MAE, RMSE, R², MAPE
- 颜色区分不同模型
- 便于快速识别最佳模型

#### 2. prediction_comparison.png
**内容**：预测结果与实际值对比图
- 时间序列折线图
- 展示最优4个模型的预测结果
- 包含实际值对照线

#### 3. shap_summary_plot.png
**内容**：SHAP特征重要性总结图
- 蜂群图展示特征影响
- Y轴：特征名称（按重要性排序）
- X轴：SHAP值（影响大小）
- 颜色：特征值高低

#### 4. shap_bar_plot.png
**内容**：SHAP重要性条形图
- 特征重要性排序
- 条形长度表示影响程度
- 便于快速识别关键特征

#### 5. shap_dependence_plots.png
**内容**：SHAP依赖关系图（前4个重要特征）
- 展示单个特征如何影响预测
- X轴：特征值
- Y轴：SHAP值
- 发现非线性关系

#### 6. training_history.png
**内容**：模型训练历史图
- LSTM和Transformer训练曲线
- 损失函数和MAE指标
- 训练集与验证集对比

---

## 🔍 文件命名规则

### 时间戳格式

```
YYYYMMDD_HHMMSS
```

**示例**：
- `20251017_143022` = 2025年10月17日 14:30:22

### 完整文件名示例

```
logs/
├── carbon_prediction_20251017_143022_detailed_report.txt
└── carbon_prediction_20251017_143022_runtime_log.txt

reports/
└── carbon_prediction_20251017_143022_report.xlsx

visualizations/
├── carbon_prediction_20251017_143022_model_performance_comparison.png
├── carbon_prediction_20251017_143022_prediction_comparison.png
├── carbon_prediction_20251017_143022_shap_summary_plot.png
├── carbon_prediction_20251017_143022_shap_bar_plot.png
├── carbon_prediction_20251017_143022_shap_dependence_plots.png
└── carbon_prediction_20251017_143022_training_history.png
```

---

## 📦 文件大小参考

| 文件类型 | 典型大小 | 说明 |
|---------|---------|------|
| detailed_report.txt | 10-50 KB | 文本报告 |
| runtime_log.txt | 5-20 KB | 运行日志 |
| report.xlsx | 100-500 KB | Excel报告（含6个工作表） |
| *.png | 50-200 KB/张 | 可视化图表 |

---

## 🔄 文件管理

### 自动清理

建议定期清理旧文件以节省空间：

```bash
# 删除7天前的文件
find outputs/ -type f -mtime +7 -delete

# 只保留最近5次运行的结果
ls -t outputs/logs/*.txt | tail -n +11 | xargs rm
```

### 备份重要结果

```bash
# 备份特定日期的结果
mkdir -p backups/20251017
cp outputs/*20251017* backups/20251017/
```

---

## ⚠️ 注意事项

### 1. 不要手动修改文件名
- 文件名包含时间戳，用于追溯运行历史
- 修改后可能无法正确关联

### 2. 定期清理旧文件
- 避免占用过多磁盘空间
- 保留重要的实验结果

### 3. Git 管理
- 输出文件已加入 `.gitignore`
- 不会被提交到版本控制
- 只保存代码和文档

---

## 📚 相关文档

- 📖 [README.md](../README.md) - 项目概览
- 📊 [使用指南](../docs/碳价格预测系统使用指南.md) - 详细操作说明
- 🔧 [配置说明](../docs/项目配置说明.md) - 配置参数详解

---

**目录说明版本**: v1.0  
**最后更新**: 2025-10-17  
**维护团队**: AI Research Team
