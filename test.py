# 测试样例：从 1 加到 100000 的和

# 方法1: 使用 for 循环
sum1 = 0
for i in range(1, 100000001):
    sum1 += i
print(f"Using for loop: Sum from 1 to 100000 is {sum1}")

# 方法2: 使用数学公式（等差数列求和公式）
sum2 = (100000 * (100000 + 1)) // 2
print(f"Using formula: Sum from 1 to 100000 is {sum2}")
