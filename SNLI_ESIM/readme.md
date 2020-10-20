LSTM on IMDB datasets

1.preprocessing.py 包括 读取数据、建立字典、建立数据集批量化数据

2.train.py 训练函数、注意模型和数据集参数

3.models LSTM模型


Running Instructions:

0) 下载数据集 SNLI_1.0
1) 读取数据、建立字典 make_dictionary()函数 in 'preprocessing/'
2) 建立glove嵌入层embedding   build_embedding_matrix()函数 in 'models/'
2) 'python train.py'
