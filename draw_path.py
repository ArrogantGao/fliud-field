import numpy as np
import matplotlib.pyplot as plt

path = np.loadtxt('path.txt')
x_path = path[:,0]
y_path = path[:,1]

plt.figure('PATH')
plt.plot(x_path, y_path)
plt.draw()  # 显示绘图
 
plt.pause(5)  #显示5秒
 
plt.savefig("path.jpg")  #保存图象
 
plt.close() 