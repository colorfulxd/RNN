#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data

def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)

#这里的batch size 有点不大一样
#但是仔细想想：比如有5000个字
#batch size = 10 那么数据会分成5000/10= 500 个字
#然后500个字呢，在时间轴上按RNN的 time step 取划分，比如有20个time step,那么
#就有25组这样的 time step。 画出图如下
#0 : [0,...19] [20,...29] ...[480,...499]
#1 : [0,...19] [20,...29] ...[480,...499]
#。。。
#10 : [0,...19] [20,...29] ...[480,...499]
#所谓的batch size是指，某个时间点上10组数据（纵向看）输入进去，训练一组参数，那么经过25轮这样的训练
#输出参数。
#返回的是[batchsize,number step * (total len/batchsize/numstep)]
#返回的是[batchsize,number step * 25]
def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    #参考老师的代码改
    #首先建立词向量，先记录我们的输出的含义方便后面查阅：
    # data 是我们所有的vocabulary 里的词的序号向量，比如【潘。。。】，根据50000个高频词的排序序号为100.
    # count 是[['UNK', 词频数量]，[潘，词频数量]]默认是由高频道低频
    # dictionary 是 这样的字典【字，序（序号）】
    # reverse dictionary 是 这样的字典【序（序号）,字】
    #应该用的是data 这个结构
    #data, count, unused_dictionary, reverse_dictionary  = build_dataset(vocabulary, 50000)
    #del vocabulary  # Hint to reduce memory.
    #同过data 取找到对应的词嵌入的128 维向量，这里需要导入训练的词向量矩阵
    sample_len = batch_size * (len(vocabulary)//batch_size)
    vo_array = np.array(vocabulary[:sample_len])
    vo_array = vo_array.reshape([batch_size,-1])
                     
    sample_len = vo_array.shape[1]
    len_count = random.randint(0,16)
    while True :
        length = num_steps + 1
        if length + len_count > sample_len :
            break
        yield vo_array[:,len_count:length + len_count]
        len_count = length + len_count
      

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
