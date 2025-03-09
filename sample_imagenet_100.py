import random

# 读取原始文件
with open('exp/imagenet_val_1k.txt', 'r') as f:
    lines = f.readlines()

# 随机采样100行
sampled_lines = random.sample(lines, 100)

# 重编号为0-99
renumbered_lines = []
for i, line in enumerate(sampled_lines):
    filename = line.split()[0]
    renumbered_lines.append(f"{filename} {i}\n")

# 写入新的文件
with open('exp/imagenet_val_100.txt', 'w') as f:
    f.writelines(renumbered_lines)

print("随机采样并重编号完成，结果保存在output.txt")
