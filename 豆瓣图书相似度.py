import jieba
from jieba.analyse import *
import xlrd
import nltk
from nltk import word_tokenize
import pandas as pd
import os
import numpy as np
from numpy import *;


def buildStopWordList(strStop): #生成停用词集合
    stopwords = set()
    strSplit = strStop.split('\n')
    for line in strSplit:
        # print(line)
        stopwords.add(line.strip())
    stopwords.add('\n')
    stopwords.add('\t')
    stopwords.add(' ')
    return stopwords


def buildWordSet(str, setStop):  #根据停用词过滤，并利用set去重
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    words = ' '.join(jieba.cut(str)).split(' ')  # 利用jieba工具进行中文分词
    setStr = set()
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        if word not in setStop:
            setStr.add(word)
    return setStr


def readTxtFile(filename, ec='UTF-8'): # UTF-8很重要！
    str=""
    with open(filename, "r", encoding=ec) as f:
        str = f.read() 
    return(str)


def cos(a,b):  #词对相似度计算
    dot_product = np.dot(a, b)#
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return  dot_product / (norm_a * norm_b)


def GetFileSim(set1, set2, sim_vetor):  # 计算两个文本的相似度
    it1 = set1
    it2 = set2
    len1 = len(set1)  # 文本1单词的个数
    len2 = len(set2)  # 文本2单词的个数
    # 初始化文本词矩阵
    sims = pd.DataFrame(np.zeros([len1, len2]))
    m = 0
    # 生成文本词矩阵
    for i in it1:

        n = 0
        for j in it2:
            if i == j:
                sims.loc[m, n] = 1
            else:
                try:
                    sims.loc[m, n] = sim_vetor.loc[i, j]
                except:
                    sims.loc[m, n] = 0
            n = n + 1
        m = m + 1
    '''
    row = sum(sims.max(axis=0))  # 行方向：单词向量的余弦矩阵，最值和
    col = sum(sims.max(axis=1))  # 列方向：单词向量的余弦矩阵，最值和
    # 把相似度全部为0的行与列统计
    row1 = list(sims.max(axis=0))
    col1 = list(sims.max(axis=1))
    row_zero = row1.count(0)
    col_zero = col1.count(0)
    # 计算平均相似度
    sim = (row + col) / (len1 + len2 - row_zero - col_zero)
    return sim
    '''
    v0 = sims.dropna().max(axis=0)
    v1 = sims.dropna().max(axis=1)
    fSim=((v0.sum()+v1.dropna().sum())/(len(v0.dropna())+len(v1.dropna())));
    return fSim


if __name__ == '__main__':
    #读取停用词列表
    szStopWordFile = r'D:\数据挖掘\实验二作业材料\DocSimDemo\my中文和符号1960.txt';
    #encoding = "unicode_escape"
    encoding = 'UTF-8'
    strStop = readTxtFile(szStopWordFile, encoding);
    setStop = buildStopWordList(strStop);

    #读取图书的书名
    title= pd.read_excel('D:\数据挖掘\豆瓣书籍.xlsx')['title']
    TitleSize = title.size  #书名的个数
    print(title)
    BookList = []   
    data = pd.read_excel('D:\数据挖掘\豆瓣书籍.xlsx')['content'] #读取所有图书的书评
    DataSize = data.size    #图书书评项的个数
    for i in range(DataSize):   #遍历所有图书书评项里的词
        SingleSet = buildWordSet(str(data[i]),setStop)  #过滤停用词
        BookList.append(SingleSet)  #过滤完的词集合保存在一个新的列表中
    print(BookList)
    total_set = set()
    for i in range(0,len(BookList)):
        total_set = total_set | BookList[i] #所有图书里书评项的词取并集得到新的集合

    #形成词矩阵
    m = len(total_set)  #所有词的数量
    data = pd.DataFrame(np.zeros([m, m]), columns=total_set, index=total_set) #初始化词矩阵
    #print(data)
    print("开始形成词矩阵！")
    L = TitleSize  #文件总个数
    print(L)
    for i in range(0, L):
        set1 = BookList[i]
        for j in range(i+1, L):
            set2 = BookList[j]
            v = list(set2)

            for w1 in set1:
                data.loc[w1][v] = data.loc[w1][v] + 1
                data.loc[w1][w1] = 0
    data = data + data.values.T;  # 加上转置
    print(data)
    data.to_csv('D:\数据挖掘\标注矩阵.csv',encoding = 'UTF-8')


    print("开始读入标注矩阵！")
    #读入标注矩阵
    sgign = pd.read_csv('D:\数据挖掘\标注矩阵.csv', index_col=0, encoding = 'UTF-8')  # index_col = 0 设置第一列为列名
    # print(sign)
    words = list(sign.index)  # 获取词汇
    L = len(words)


    #读入词向量表
    f = open(r'D:\数据挖掘\实验二作业材料\DocSimDemo\corpus\100000-small.txt', "r", encoding='UTF-8')

    lines = f.readlines()
    #词向量字典
    vectors = {}
    for line in lines:
        # 分离出向量
        value = list(map(float, line.split()[1:]))  # 转换为浮点型
        # {词}='向量'
        vectors[line.split()[0]] = value
    #构建相似度矩阵
    sim_vector = pd.DataFrame(np.zeros([L, L]), columns=words, index=words)

    for i in range(0, L - 1):
        for j in range(i + 1, L):
            # 遍历词对
            if sign.iloc[i, j] != 0:  # 词对值在标注矩阵里不为0
                try:
                    # 计算词对相似度
                    sim = cos(vectors[words[i]], vectors[words[j]])
                    # print (sim)
                    # 写入词相似度矩阵
                    sim_vector.iloc[i, j] = sim
                except:
                    next

    sim_vector = sim_vector + sim_vector.values.T
    print('相似度矩阵构造完成！')
    sim_vetor.to_csv('D:\数据挖掘\相似度矩阵.csv', encoding='UTF-8')

    title = list(title)
    #构建图书相似度矩阵
    #print(title)
    sim_vetor = pd.read_csv('D:\数据挖掘\相似度矩阵.csv', index_col=0, encoding = 'UTF-8')
    FileSim = pd.DataFrame(np.zeros([TitleSize, TitleSize]),columns = title, index = title)  # 文件之间的相似矩阵
    print("开始构建图书相似度矩阵！")
    for i in range(0, TitleSize):
        print(i)
        for j in range(i + 1, TitleSize):
            FileSim.iloc[i, j] = GetFileSim(BookList[i], BookList[j], sim_vetor)
    print(FileSim)
    FileSim.to_csv('D:\数据挖掘\图书相似度矩阵.csv', encoding="ANSI")

    # 画热点图, cmap控制颜色画图风格
    import matplotlib.pyplot as plt
    import seaborn as sns;
    import matplotlib.font_manager as fm

    file_sim.index = title
    file_sim.columns = title
    sns.set(font='STSong')  # 解决中文字体显示
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(file_sim, annot=True, linewidths=.5, ax=ax, cmap="vlag")
    plt.show();

    print("结束了？")
