#
# from collections import defaultdict, Counter
# import random
# from collections import OrderedDict
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# from scipy.spatial.distance import pdist
# from scipy.spatial.distance import squareform
# from torch_geometric.nn import SAGEConv, GCNConv, TransformerConv, MixHopConv
# import torch
# import torch_geometric
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import coo_matrix
# import torch.nn.functional as F
# import seaborn as sns
# from torch_geometric.loader import GraphSAINTNodeSampler, GraphSAINTRandomWalkSampler
# import torch
# import torchvision.ops.focal_loss as focal_loss
#
# import warnings
# warnings.filterwarnings("ignore")
#
# device = torch.device('cuda:0')
#
# # 抽取源节点（少数类）和目标节点
# def sampling_idx_individual_dst(class_num_list, idx_info):
#
#     max_num, n_cls = max(class_num_list), len(class_num_list)
#     sampling_list = max_num * torch.ones(n_cls) - torch.tensor(class_num_list)  # 顺序显示每一类需要抽取多少才能和多数类数据数量一致
#     new_class_num_list = torch.Tensor(class_num_list)
#
#     # 对每个类别的索引列表进行随机采样，采样数量由 sampling_list 中对应的值确定（可以重复抽取）
#     sampling_src_idx = [random.choices(cls_idx, k=int(samp_num.item()))
#                         for cls_idx, samp_num in zip(idx_info, sampling_list)]
#     sampling_src_idx = [sublist for sublist in sampling_src_idx if sublist]  # 移除空列表
#     sampling_src_idx = [[x] if isinstance(x, int) else x for inner_list in sampling_src_idx for x in inner_list]
#     sampling_src_idx = torch.Tensor(sampling_src_idx)
#
#     # 抽取目标节点作为邻居
#     prob = torch.log(new_class_num_list.float()) / new_class_num_list.float() # 每个类别的采样概率
#     prob = prob.repeat_interleave(new_class_num_list.long())
#     temp_idx_info = torch.cat([torch.tensor(cls_idx) for cls_idx in idx_info])
#     dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)  # 从概率分布 prob 中抽取目标节点的索引
#     sampling_dst_idx = temp_idx_info[dst_idx]
#
#     # Sorting src idx with corresponding dst idx
#     sampling_dst_idx = sampling_dst_idx.unsqueeze(1)
#
#     return sampling_src_idx, sampling_dst_idx
#
# # 寻找节点邻居
# def node_neighbors(node_idx, data_edge_index):
#     neighbors_node = []
#     for node_index in node_idx:
#         edges = (data_edge_index == node_index).any(dim=0)
#         # 找到该节点的邻居节点
#         neighbors0 = data_edge_index[0, edges].tolist()
#         neighbors1 = data_edge_index[1, edges].tolist()
#         neighbors = neighbors0 + neighbors1
#         neighbors = list(filter(lambda x: x != node_index, neighbors))  # 去除节点本身
#         neighbors = list(set(neighbors))
#         neighbors_node.append(neighbors)
#
#     return neighbors_node
#
# # 构造新邻居
# def node_edge(neighbors, idx):
#     new_edge = []
#     idx = torch.tensor(idx)
#     # 寻找数据的无值的索引
#     empty_indices = [i for i, sublist in enumerate(neighbors) if not sublist]
#     # 去除对应索引的值
#     values = [value for i, value in enumerate(idx) if i not in empty_indices]
#     neighbors_values = [value for i, value in enumerate(neighbors) if i not in empty_indices]
#     # 按照邻居个数改变源节点数目
#     list = [tensor_item.repeat(len(sublist)).tolist() for tensor_item, sublist in zip(values, neighbors_values)]
#     # 将嵌套列表变成一维
#     list = [item for sublist in list for item in sublist]
#     neighbors_list = [item for sublist in neighbors_values for item in sublist]
#
#     new_edge.append(list)
#     new_edge.append(neighbors_list)
#     # print(new_edge)
#
#     return new_edge
#
#
# class GNN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_channels):
#         super(GNN, self).__init__()
#         self.conv1 = SAGEConv(num_features, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = x.to(torch.float32)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x0 = self.conv2(x, edge_index)
#         x = F.softmax(x0, dim=1)  # dim=1表示对每一行进行运算，最终每一行之和加起来为1
#         return x, x0
#
#
# print("start")
# # 构建新的临阶矩阵
# data_train = pd.read_csv("code_train.csv", header=0, skiprows=0)
# data_train_y = pd.DataFrame(data_train.label)
# data_train_y = list(data_train_y.T.loc['label'])
#
# data_train.drop(columns=['label'], inplace=True)
# node_features = torch.tensor(data_train.values)
#
# edge_index = pd.read_csv('edge_train.csv', header=0, skiprows=0)
# edge_index = torch.tensor(edge_index.T.values)
#
# # 统计每一类数据的数目
# # 使用 Counter 统计每个元素的个数
# counter = Counter(data_train_y)
# # 按元素的顺序输出个数
# class_num_list = [counter[i] for i in sorted(set(data_train_y))]
# # print(class_num_list)  # [10000, 411, 10000, 8, 10000, 2905, 458] eg:0类有10000个数据
#
# # 构建类索引字典
# idx_info = defaultdict(list)
# for i, label in enumerate(data_train_y):
#     idx_info[label].append(i)
# # 将 defaultdict 转换为普通字典，并按类别从小到大排序
# # print(idx_info) # {5: [0, 1, 2],6:...}
# idx_info = dict(sorted(idx_info.items()))
# idx_info = idx_info.values()
# # print("length,", len(idx_info), 'idx_info,', idx_info)
#
# sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info)
# sampling_src_idx = sampling_src_idx.long()
# print('sampling_src_idx', sampling_src_idx)
# print('sampling_dst_idx', sampling_dst_idx)
#
# dst_neighbors = node_neighbors(sampling_dst_idx, edge_index)
# src_neighbors = node_neighbors(sampling_src_idx.long(), edge_index)
# print('dst_neighbors', dst_neighbors)
# print('src_neighbors', src_neighbors)
#
#
# # 添加新节点的标签值
# data_train_y = torch.tensor(data_train_y)
# selected_labels = torch.gather(data_train_y, 0, sampling_src_idx.squeeze(dim=1))
# new_data_train_y = torch.cat([data_train_y, selected_labels], dim=0)
# # print('new_y', new_data_train_y)
#
# new_y0 = pd.DataFrame(new_data_train_y)
# new_y0.to_csv("new_y_1.csv", index=False, header=1)
#
# # 将sampling_src_idx中的索引重新编号
# result_indices = list(range(len(node_features), len(node_features) + len(sampling_src_idx)))
# # print(result_indices)
#
# # 构建新的边数据集
# new_edge0 = node_edge(dst_neighbors, result_indices)
# new_edge1 = node_edge(src_neighbors, result_indices)
# new_edge2 = [sublist1 + sublist2 for sublist1, sublist2 in zip(new_edge0, new_edge1)]
# # print(new_edge2)
# # 拼接到原有数据后面
# edge_index0 = edge_index.squeeze().tolist()
# new_edge = [sublist1 + sublist2 for sublist1, sublist2 in zip(edge_index0, new_edge2)]
# # print(new_edge)
# new_edge = torch.tensor(new_edge)
#
# new_edge0 = pd.DataFrame(new_edge)
# new_edge0.to_csv("new_edge_1.csv", index=False, header=1)
#
# # 构建新的特征矩阵   内积系数
# # 寻找每个节点的邻居数目
# num_neighbors = []
# for i in result_indices:
#     occurrences = torch.sum(torch.eq(new_edge[0], i))
#     num_neighbors.append(occurrences)
# # print(num_neighbors)
# n = 0
# new_node_feature = node_features
# for i in sampling_src_idx:
#     print(i)
#     sampling_src_idx_feature = node_features[i]
#
#     query_feature = sampling_src_idx_feature.repeat((int(num_neighbors[n]+1), 1))
#
#     dst_neighbors_feature = node_features[dst_neighbors[n]]
#     src_neighbors_feature = node_features[src_neighbors[n]]
#     value_feature = torch.cat([dst_neighbors_feature, src_neighbors_feature], dim=0)
#     value_feature = torch.cat([value_feature, sampling_src_idx_feature], dim=0)
#
#     d_k = query_feature.size(-1)
#     attention_scores = torch.matmul(query_feature, value_feature.t()) / np.sqrt(d_k)
#     attention_probs = nn.Softmax(dim=-1)(attention_scores)
#     context_layer = torch.matmul(attention_probs, value_feature)
#
#     new_node_feature = torch.cat([new_node_feature, context_layer[0].unsqueeze(0)], dim=0)
#
#
# print(new_node_feature.shape)
# new_node_feature0 = pd.DataFrame(new_node_feature)
# new_node_feature0.to_csv("new_node_feature_1.csv", index=False, header=1)
#
# new_edge = pd.read_csv("new_edge_1.csv", header=0, skiprows=0)
# new_edge = np.array(new_edge)
# new_edge = torch.tensor(new_edge, dtype=torch.long)
# print(new_edge)
#
# new_node_feature = pd.read_csv("new_node_feature_1.csv", header=0, skiprows=0)
# new_node_feature = np.array(new_node_feature)
# new_node_feature = torch.tensor(new_node_feature, dtype=torch.float)
# print(new_node_feature)
#
# new_data_train_y = pd.read_csv("new_y_1.csv", header=0, skiprows=0)
# new_data_train_y = np.array(new_data_train_y)
# new_data_train_y = torch.tensor(new_data_train_y)
# new_data_train_y = new_data_train_y.view(-1)
# print(new_data_train_y)
#
# # 构建图数据集
# graph = torch_geometric.data.Data(x=new_node_feature, edge_index=new_edge, y=new_data_train_y)
#
# # 实现GraphSAINT
# loader_train = GraphSAINTRandomWalkSampler(data=graph, batch_size=10000, walk_length=6, num_steps=7)
#
# # 构建模型
# model = GNN(num_features=78, num_classes=7, hidden_channels=160)
# Epoch = 200
# train_losses = []
# accuracy_train = []
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()
# num_batches = len(loader_train)
# print('num_batches', num_batches)
# total_x = []
# total_x0 = []
# for epoch in range(Epoch):
#     print('epoch', epoch)
#     model.train()
#     total_correct = 0
#     total_samples = 0
#     losses = 0
#     if epoch == Epoch-1:
#         for data in loader_train:
#             x, x0 = model(data)
#             total_x.append(data.x.tolist())
#             total_x0.append(x0.tolist())
#
#     for data in loader_train:
#         x, x0 = model(data)
#         # loss_1 = torch.nn.CrossEntropyLoss()(x, data.y)
#         loss_1 = focal_loss(x, data.y)
#         loss_total = loss_1
#
#         losses += float(loss_total.item())
#         predicted = x.argmax(dim=1)
#         total_samples += len(data.y)
#         total_correct += sum(x == y for x, y in zip(data.y, predicted))
#
#         optimizer.zero_grad()
#         loss_total.backward()
#         optimizer.step()
#     train_losses.append(losses / num_batches)
#     print('loss', losses / num_batches)
#     # 计算每个 epoch 的训练精度
#     accuracy_train.append(total_correct / total_samples)
#     print('accuracy_train', total_correct / total_samples)
#
# model_name = "my_model_SAGE.pt"
# torch.save(model, model_name)
#
# train_losses = pd.DataFrame(train_losses)
# train_losses.to_csv('train_losses_SAGE.csv', header=False, index=False)
#
# accuracy_train = pd.DataFrame(accuracy_train)
# accuracy_train.to_csv('accuracy_train_SAGE.csv', header=False, index=False)
#
# total_x = pd.DataFrame(total_x)
# total_x.to_csv('train_x_SAGE.csv', header=False, index=False)
#
# total_x0 = pd.DataFrame(total_x0)
# total_x0.to_csv('train_x0_SAGE.csv', header=False, index=False)



# 娴嬭瘯
import pandas as pd
from torch_geometric.nn import SAGEConv, GCNConv, TransformerConv, MixHopConv
import torch_geometric
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch.nn.functional as F
import torch
import math
import numpy as np


import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:0')

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x0 = self.conv2(x, edge_index)
        x = F.softmax(x0, dim=1)  # dim=1琛ㄧず瀵规瘡涓€琛岃繘琛岃繍绠楋紝鏈€缁堟瘡涓€琛屼箣鍜屽姞璧锋潵涓?
        return x, x0

def lg(data):
    data['SourceIP'] = data.SourceIP.apply(str)
    data['SourcePort'] = data.SourcePort.apply(str)
    data['DestinationIP'] = data.DestinationIP.apply(str)
    data['DestinationPort'] = data.DestinationPort.apply(str)

    data['Source'] = data['SourceIP'] + ':' + data['SourcePort']
    data['Destination'] = data['DestinationIP'] + ':' + data['DestinationPort']

    data.drop(columns=['SourcePort', 'DestinationPort', 'SourceIP', 'DestinationIP'], inplace=True)

    data['e'] = range(0, len(data))

    df = pd.DataFrame(data=None, columns=['s', 'd'])
    for i in range(len(data)):
        index = data[data['Source'] == data['Destination'][i]].index.tolist()
        s = [data['e'][i]] * len(index)
        s = pd.DataFrame(s)
        s.reset_index(inplace=True, drop=True)
        d = pd.DataFrame(data['e'][index])
        d.reset_index(inplace=True, drop=True)

        if len(index) != 0:
            df0 = pd.concat([s, d['e']], axis=1)
            df0.columns = ['s', 'd']
            df = pd.concat([df, df0], axis=0)
    return df


print("start")

datas = pd.read_csv("code_test.csv", chunksize=10000, header=0, skiprows=0)
all_batch_predictions = []
all_label = []
# 鍔犺浇妯″瀷
model_name = "my_model_SAGE.pt"
model = torch.load(model_name)
model.eval()
with torch.no_grad():
    for test in datas:
        test = test.reset_index(drop=True)
        y = list(test.T.loc['label'])
        y = torch.tensor(y)
        y = y.view(-1)

        node_features = test.drop(columns=['SourcePort', 'DestinationPort', 'SourceIP', 'DestinationIP', 'label'],
                                  inplace=False)
        node_features = torch.tensor(node_features.values)

        edge = lg(test)
        edge = edge.T
        edge = edge.astype(int)
        edge = torch.tensor(edge.values, dtype=torch.long)

        graph = torch_geometric.data.Data(x=node_features, edge_index=edge, y=y)

        output, _ = model(graph)
        prediction = output.argmax(dim=1)

        all_batch_predictions.append(prediction.cpu().tolist())
        print('all_batch_predictions', all_batch_predictions)
        all_label.append(graph.y.cpu().tolist())
        print("all_label", len(all_label))

all_batch_predictions = pd.DataFrame(all_batch_predictions)
all_batch_predictions.to_csv('pred_SAGE.csv', header=False, index=False)

all_label = pd.DataFrame(all_label)
all_label.to_csv('y_SAGE.csv', header=False, index=False)


