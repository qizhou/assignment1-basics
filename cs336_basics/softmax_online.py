# a simple demo for online softmax trick:
# evaluate softmax with
#   z = sum s_i * y_i
# where s_i = softmax(x_i)
# in practice, y is the vector in V
import random
import math

n = 8
x = [random.normalvariate() for _ in range(n)]
y = [random.normalvariate() for _ in range(n)]

# Classic method, substract max(x) to avoid numerical issue
# which needs two pass
max_x = max(x)
exp_x = [math.exp(xi - max_x) for xi in x] # stable version
sum_exp = sum(exp_x)
s = [ex / sum_exp for ex in exp_x]
z = sum([s[i] * y[i] for i in range(n)])

print("z", z)

# Online method with one pass
m = 0
l = 0
z1 = 0
o = 0
for i in range(n):
    if i == 0:
        m = m_new = x[i]
    else:
        m_new = max(m, x[i])
    l_new = l * math.exp(m - m_new) + math.exp(x[i] - m_new)
    z1 = math.exp(x[i] - m_new) / l_new * y[i] + z1 * math.exp(m - m_new) * l / l_new
    o = math.exp(x[i] - m_new) * y[i] + o * math.exp(m - m_new)
    m = m_new
    l = l_new
print("z1", z1)
print("z2", o / l)
