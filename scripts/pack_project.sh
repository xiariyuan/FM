#!/bin/bash
# 打包FMtrack项目（排除模型权重、输出、依赖等）

set -e

PROJECT_NAME="FMtrack_project_$(date +%Y%m%d_%H%M%S)"
ZIP_FILE="${PROJECT_NAME}.zip"

echo "======================================"
echo "打包FMtrack项目"
echo "======================================"

# 创建zip，排除不需要的文件和目录
zip -r "$ZIP_FILE" . \
  -x "weight/*" \
  -x "outputs/*" \
  -x "*.whl" \
  -x "*.so" \
  -x "*.egg-info/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*/*/__pycache__/*" \
  -x "*/*/*/__pycache__/*" \
  -x "build/*" \
  -x "log/*" \
  -x "logs/*.log" \
  -x ".cache/*" \
  -x ".claude/*" \
  -x "*.zip" \
  -x "nohup.out" \
  -x ".ipynb_checkpoints/*" \
  -x "*/.ipynb_checkpoints/*" \
  -x ".git/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.pth" \
  -x "*.pt" \
  -x "*.ckpt" \
  -x "*.pdf"

echo ""
echo "======================================"
echo "打包完成！"
echo "文件名: $ZIP_FILE"
echo "大小: $(du -h $ZIP_FILE | cut -f1)"
echo "======================================"
echo ""
echo "包含的主要内容："
echo "  ✓ 所有Python代码"
echo "  ✓ configs/ 配置文件"
echo "  ✓ models/ 模型定义"
echo "  ✓ scripts/ tools/ 工具脚本"
echo "  ✓ datasets/ data/ 数据处理代码"
echo "  ✓ TrackEval/ 评估代码"
echo "  ✓ README.md 等文档"
echo "  ✓ requirements.txt 依赖列表"
echo ""
echo "排除的内容："
echo "  ✗ weight/ (模型权重)"
echo "  ✗ outputs/ (输出文件)"
echo "  ✗ *.whl *.so (依赖包)"
echo "  ✗ __pycache__/ build/ (缓存和编译产物)"
echo "  ✗ logs/*.log (日志文件)"
echo "  ✗ *.pth *.pt *.ckpt (权重文件)"
echo "======================================"
