# This is an example py file 
# By ChatGPT

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_y = 0.0

    def filter(self, x):
        y = self.alpha * x + (1 - self.alpha) * self.prev_y
        self.prev_y = y
        return y

# 使用示例
lpf = LowPassFilter(alpha=0.1)
data = [1, 2, 3, 4, 5]  # 输入数据
filtered_data = [lpf.filter(x) for x in data]
print(filtered_data)

class HighPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_x = 0.0
        self.prev_y = 0.0

    def filter(self, x):
        y = self.alpha * (self.prev_y + x - self.prev_x)
        self.prev_x = x
        self.prev_y = y
        return y

# 使用示例
hpf = HighPassFilter(alpha=0.1)
data = [1, 2, 3, 4, 5]  # 输入数据
filtered_data = [hpf.filter(x) for x in data]
print(filtered_data)
