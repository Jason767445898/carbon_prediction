# 🌍 AI碳价格预测系统

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 基于深度学习和可解释性AI的综合碳价格预测分析系统

## 📋 项目概述

本项目是一个先进的碳价格预测分析工具，整合了多种最新的机器学习和深度学习技术。系统通过智能的特征工程、强大的时间序列建模和透明的可解释性分析，为碳交易市场提供精准的价格预测和决策支持。

### 🎯 核心功能

- **多模型预测**：集成 LSTM、Transformer、随机森林等多种预测模型
- **智能预处理**：自动化数据清洗、特征工程和技术指标生成
- **可解释分析**：基于 SHAP 的模型解释性分析，满足监管要求
- **可视化输出**：生成专业的分析报告和可视化图表
- **完整追溯**：详细的运行日志和配置管理

### 🔧 技术架构

| 技术组件 | 用途 | 核心优势 |
|---------|------|----------|
| **LSTM** | 时间序列预测 | 捕捉长期依赖关系 |
| **Transformer** | 复杂模式识别 | 并行计算，全局建模 |
| **SHAP** | 模型可解释性 | 提供透明的决策依据 |
| **机器学习模型** | 基准对比分析 | 稳定的预测基线 |

## 🚀 快速开始

### 环境要求

- Python 3.7+
- 推荐使用 Anaconda 环境管理

### 安装依赖

```bash
# 基础科学计算库
pip install numpy pandas matplotlib seaborn

# 机器学习库
pip install scikit-learn tensorflow

# 高级模型库
pip install xgboost lightgbm

# 可解释性和数据处理
pip install shap openpyxl
```

### 一键运行

```python
from carbon_price_prediction import CarbonPricePredictionSystem

# 使用示例数据快速体验
system = CarbonPricePredictionSystem()
system.run_complete_analysis()

# 使用自己的数据
system = CarbonPricePredictionSystem()
system.run_complete_analysis('your_data.xlsx')
```

## 📊 数据要求

### 基本格式要求

您的数据文件应包含以下基本结构：

| 日期 | carbon_price | 影响因子1 | 影响因子2 | ... |
|------|--------------|-----------|-----------|-----|
| 2020-01-01 | 50.25 | 2.1 | 65.3 | ... |
| 2020-01-02 | 51.10 | 2.1 | 66.1 | ... |

### 数据准备清单

- ✅ 文件格式：`.xlsx`、`.csv` 或 `.dta` (Stata文件)
- ✅ 时间列：连续的日期数据
- ✅ 目标列：碳价格数据
- ✅ 特征列：相关影响因子（建议5-15个）
- ✅ 数据量：建议至少500个数据点
- ✅ 数据质量：缺失值 < 5%

### Stata文件支持

系统现在支持Stata (.dta) 文件格式。有关如何使用Stata文件的详细信息，请参阅 [STATA_USAGE_GUIDE.md](./STATA_USAGE_GUIDE.md)。

## 📈 输出结果

运行系统后，将自动生成以下文件：

### 📊 分析报告
- `carbon_prediction_*_detailed_report.txt` - 详细分析报告
- `carbon_prediction_*_runtime_log.txt` - 运行日志
- `carbon_prediction_*_report.xlsx` - Excel综合报告

### 📈 可视化图表
- `*_model_performance_comparison.png` - 模型性能对比图
- `*_prediction_comparison.png` - 预测结果对比图
- `*_shap_summary_plot.png` - SHAP特征重要性总结图
- `*_shap_bar_plot.png` - SHAP重要性条形图
- `*_shap_dependence_plots.png` - SHAP依赖关系图
- `*_training_history.png` - 模型训练历史图

### 📋 数据表格
- Excel报告包含6个工作表：数据概要、模型性能、特征重要性、预测结果、系统配置、数据质量

## ⚙️ 配置说明

### 基础配置
```python
config = {
    'target_column': 'carbon_price',  # 目标列名
    'sequence_length': 60,            # 时间序列长度
    'test_size': 0.2,                # 测试集比例
}
```

### 模型参数
```python
config = {
    'lstm_config': {
        'units': [64, 32],     # 网络结构
        'epochs': 100,         # 训练轮数
        'batch_size': 32       # 批处理大小
    },
    'transformer_config': {
        'd_model': 128,        # 模型维度
        'num_heads': 8,        # 注意力头数
        'epochs': 50           # 训练轮数
    }
}
```

## 📚 文档结构

```
📁 AI/
├── 📄 README.md                      # 项目概览（本文件）
├── 📄 碳价格预测系统使用指南.md         # 详细使用指南
├── 🐍 carbon_price_prediction.py     # 主程序文件
├── 📁 model_knowledge/               # 技术知识库
│   ├── 📄 1_LSTM金融预测教程.md
│   ├── 📄 2_Transformer_Attention机制教程.md
│   ├── 📄 3_SHAP可解释性分析教程.md
│   ├── 📄 4_EMD经验模态分解教程.md
│   ├── 📄 5_综合金融预测示例.md
│   └── 📄 AI金融预测知识点分析.md
├── 📁 carbon_prediction_log_txt/     # 运行日志目录
├── 📁 carbon_prediction_results_excel/ # Excel报告目录
└── 📁 carbon_prediction_results_pic/   # 图表目录
```

## 🔍 核心特性

### 🎯 多模型集成
- **深度学习模型**：LSTM、Transformer
- **机器学习模型**：随机森林、梯度提升、XGBoost
- **智能集成**：自动选择最优模型组合

### 🧠 智能特征工程
- 自动生成技术指标（RSI、布林带、移动平均等）
- 价格动量和波动率特征
- 滞后特征和交互特征

### 📊 可解释性分析
- SHAP值分析，量化每个特征的贡献
- 特征重要性排序
- 决策过程可视化

### 🎨 专业可视化
- 模型性能对比图表
- 预测结果时间序列图
- SHAP分析图表
- 训练过程监控图

### 🛠️ 高级用法

### 分步执行
```python
system = CarbonPricePredictionSystem()
system.load_data('data.xlsx')
system.preprocess_data()
system.train_models()
system.evaluate_models()
system.perform_shap_analysis()
system.create_visualizations()
system.generate_report()
```

### 自定义配置
```python
custom_config = {
    'target_column': '你的碳价格列名',
    'sequence_length': 90,
    'lstm_config': {
        'units': [128, 64],
        'epochs': 150
    }
}
system = CarbonPricePredictionSystem(config=custom_config)
```

### 批量分析
```python
datasets = ['data_2020.xlsx', 'data_2021.xlsx', 'data_2022.xlsx']
for dataset in datasets:
    system = CarbonPricePredictionSystem()
    system.run_complete_analysis(dataset)
```

## 📊 性能指标

系统提供以下评估指标：

- **MSE/MAE/RMSE**：预测误差指标，越小越好
- **R²**：决定系数，越接近1越好
- **MAPE**：平均绝对百分比误差，越小越好
- **方向准确率**：预测趋势的准确性，越高越好

### 性能基准
- **R² > 0.8**：模型性能优秀，可用于实际预测
- **R² > 0.6**：模型性能良好，建议继续优化  
- **方向准确率 > 70%**：趋势预测可靠

### 📈 特征重要性分析
系统自动生成以下技术指标：
- 价格变化特征（收益率、价格差值）
- 移动平均特征（5、1020、30天）
- 波动率特征（7、14、30天）
- 技术指标（RSI、布林带、动量）
- 滚后特征（1-10天）

## 🔧 系统要求

### 硬件要求
- **内存**：建议 8GB 以上
- **存储**：至少 2GB 可用空间
- **处理器**：支持多核并行计算
- **GPU**：可选，但推荐用于深度学习模型训练

### 软件环境
- **操作系统**：Windows/macOS/Linux
- **Python**：3.7 或更高版本
- **包管理**：pip 或 conda

## ⚠️ 注意事项

- 确保数据质量和完整性
- 深度学习模型需要较长训练时间
- SHAP分析可能比较耗时
- 建议在GPU环境中运行大型模型

## 📞 获取帮助

### 快速故障排除
1. **安装问题**：查看 [使用指南](./碳价格预测系统使用指南.md) 的环境配置部分
2. **数据问题**：参考使用指南的数据格式要求
3. **性能问题**：调整配置参数或使用更小的模型

### 学习资源
- 📖 [详细使用指南](./碳价格预测系统使用指南.md) - 完整的使用说明
- 🧠 [技术知识库](./model_knowledge/) - 深入理解核心技术
- 💡 [最佳实践案例](./model_knowledge/5_综合金融预测示例.md) - 实际应用示例

## 🤝 贡献指南

欢迎提交问题反馈和改进建议！

## 📋 项目状态

- **稳定版本**: v1.0
- **最后更新**: 2025年1月  
- **代码质量**: 完整测试，生产就绪
- **文档状态**: 与代码实现完全同步
- **支持的Python版本**: 3.7 - 3.11
- **测试环境**: Windows 10/11, macOS 12+, Ubuntu 20.04+

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**快速链接**：
- 📖 [详细使用指南](./碳价格预测系统使用指南.md) 
- 🔧 [环境安装指南](./碳价格预测系统使用指南.md#系统要求)
- 📊 [数据准备指南](./碳价格预测系统使用指南.md#数据格式要求)
- 🛠️ [故障排除](./碳价格预测系统使用指南.md#故障排除)
