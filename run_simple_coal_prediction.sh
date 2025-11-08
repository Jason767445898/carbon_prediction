#!/bin/bash
# 数据优化版煤炭价格预测运行脚本（防休眠）

echo "========================================="
echo "数据优化版 LSTM + Attention 煤炭价格预测"
echo "========================================="
echo ""
echo "数据处理优化:"
echo "  ✓ 增强特征工程 (80+ 技术指标)"
echo "  ✓ Robust Scaler 标准化"
echo "  ✓ MAD 异常值移除"
echo "  ✓ 互信息特征选择"
echo "  ✓ 时间周期性编码"
echo ""
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 使用 caffeinate 防止系统休眠
caffeinate -i python3 simple_lstm_attention_coal.py

echo ""
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "请查看 outputs/ 目录下的分析结果"
echo "========================================="
