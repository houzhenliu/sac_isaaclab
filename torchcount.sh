#!/bin/bash
# 文件名: 
# 用法: ./torchcount.sh tensors32000

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tensor_sizes_file>"
    exit 1
fi

INPUT_FILE=$1

# 用 Python 统计相同 shape 的数量
python3 - <<END
import re
from collections import Counter

with open("$INPUT_FILE", "r") as f:
    lines = f.readlines()

shapes = []
for line in lines:
    match = re.search(r'torch\.Size\((\[.*?\])\)', line)
    if match:
        shapes.append(match.group(1))

counter = Counter(shapes)

for shape, count in counter.most_common():
    print(f"{shape}: {count}")
END