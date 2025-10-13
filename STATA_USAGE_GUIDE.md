# Stata文件使用指南

本文档说明如何在碳价格预测系统中使用Stata (.dta) 文件。

## 支持的文件格式

碳价格预测系统现在支持以下文件格式：
- Excel文件 (.xlsx, .xls)
- CSV文件 (.csv)
- Stata文件 (.dta)

## 如何使用Stata文件

### 1. 直接使用Stata文件

```python
from carbon_price_prediction import CarbonPricePredictionSystem

# 创建系统实例
system = CarbonPricePredictionSystem()

# 直接使用Stata文件进行分析
system.run_complete_analysis('your_data.dta')
```

### 2. 转换现有文件为Stata格式

如果已有Excel或CSV文件，可以转换为Stata格式：

```python
import pandas as pd

# 读取现有数据
data = pd.read_excel('your_data.xlsx')  # 或 pd.read_csv('your_data.csv')

# 保存为Stata文件
data.to_stata('your_data.dta', write_index=False)
```

### 3. 数据要求

Stata文件应满足以下要求：
1. 第一列应为日期列（系统会自动将其设置为索引）
2. 包含碳价格列（默认列名为`carbon_price`，系统会自动识别相似列名）
3. 包含相关影响因子（如GDP、工业生产指数、能源价格等）
4. 数据应按时间顺序排列

## 示例数据结构

一个典型的Stata文件应包含如下列：
- `date`: 日期（第一列）
- `carbon_price`: 碳价格（目标变量）
- `gdp_growth`: GDP增长率
- `industrial_production`: 工业生产指数
- `oil_price`: 石油价格
- `gas_price`: 天然气价格
- 其他相关经济指标...

## 注意事项

1. 确保pandas版本支持Stata文件读取功能
2. Stata文件中的日期列会被自动识别并设置为索引
3. 如果日期列无法自动解析，系统会尝试将其转换为日期格式
4. 系统会自动识别碳价格列，即使列名略有不同（如`carbon_price`, `price`, `碳价格`等）

## 故障排除

如果遇到问题：
1. 检查pandas版本：`pandas.read_stata`功能需要pandas 0.18.0或更高版本
2. 确保Stata文件未被其他程序占用
3. 检查文件路径是否正确
4. 确认文件未损坏

## 联系支持

如有任何问题，请联系技术支持团队。