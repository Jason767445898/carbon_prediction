#!/bin/bash
# 煤炭价格预测 - 第一轮参数优化执行脚本
# 防止系统休眠并执行优化任务

echo "=========================================="
echo "  煤炭价格预测 - 第一轮参数优化"
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "配置数量: 20"
echo "预计耗时: 约60-90分钟"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"
echo ""

# 切换到项目目录
cd "$(dirname "$0")/.." || exit 1

echo "📂 工作目录: $(pwd)"
echo ""

# 检查优化脚本是否存在
if [ ! -f "parameter/coal_price_optimization_round1.py" ]; then
    echo "❌ 错误: 未找到优化脚本"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "data.dta" ]; then
    echo "❌ 错误: 未找到数据文件 data.dta"
    exit 1
fi

echo "🚀 启动优化任务..."
echo "   使用 caffeinate 防止系统休眠"
echo ""

# 使用 caffeinate 防止休眠，并执行优化脚本
caffeinate -i python3 parameter/coal_price_optimization_round1.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 优化任务成功完成!"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    echo "📊 结果文件保存在 outputs/optimization_round1/ 目录下"
else
    echo ""
    echo "=========================================="
    echo "❌ 优化任务执行失败"
    echo "请检查日志输出"
    echo "=========================================="
    exit 1
fi
