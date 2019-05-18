import Levenshtein
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


#进行文本文件的空格补偿
def pad_list(text_list: list, size: int) -> int:
    for i in range(len(text_list),size):
        text_list.append(' ')
    return text_list

#计算文本文件每个单词对的编辑距离的均值
def average_edit_distance(a: str, b: str) -> float:
    a_list, b_list = a.split(' '), b.split(' ')
    text_size = max(len(a), len(b))
    a_list, b_list = pad_list(a_list, text_size), pad_list(b_list, text_size)
    s = np.zeros(text_size)
    for i in range(text_size):
        s[i] = Levenshtein.distance(a_list[i], b_list[i])
    return np.average(s)


#计算文本文件整体的编辑距离
def total_edit_distance(a: str, b: str) -> float:
    return Levenshtein.distance(a,b)


#数据可视化：生成二维数组的热力图
def heatmap(s: [])
    ax = sns.heatmap(s, annot=True, fmt='.2f', linewidths=.5, square=True, cmap='YlGnBu')
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    ax.set_title('EditDistance Heatmap', fontsize=18)
    plt.show()


def main():
    n = int(input("请输入要计算的文本数量："))
    start = time.perf_counter()
    s = np.zeros((n, n))
    for i in range(n):
        f = open(r"C:\Users\kaika\Desktop\数据挖掘实验一data\data\\" + str(i + 1) + ".txt")
        a = f.read()
        for j in range(n):
            f = open(r"C:\Users\kaika\Desktop\数据挖掘实验一data\data\\" + str(j + 1) + ".txt")
            b = f.read()
            s[i][j] = total_edit_distance(a, b)     #计算文本文件整体的编辑距离
            ##s[i][j] = average_edit_distance(a, b)     #计算文本文件每个单词对的编辑距离的均值
    result = ''
    for i in range(n):
        result += ' '.join([str(x) for x in s[i]]) + '\n'
    with open('result.txt', 'w') as f:
        f.write(result)
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
    heatmap(s)


if __name__ == '__main__':
    main()
