import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
import copy

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def load_blog_catalog(select_ids):

    adj_lists = defaultdict(set)
    adj_lists_empty = defaultdict(set)

    with open("../BlogCatalog-data/bc_adjlist.txt", "r") as fp:
        for line in fp:
            vals = line.split(" ")
            for x in vals[1:]:
                adj_lists[int(vals[0])].add(int(x))
                if int(vals[0]) not in select_ids and int(x) not in select_ids:
                    adj_lists_empty[int(vals[0])].add(int(x))

    num_nodes = 10312
    num_feats = 128
    features = np.zeros((num_nodes, num_feats))
    with open("../BlogCatalog-data/vec_all.txt", "r") as fp:
        for lines in fp:
            line = lines.strip().split(" ")
            ind = 0
            for x in line[1:]:
                features[int(line[0])][ind] = float(x)
                ind += 1
    return adj_lists, adj_lists_empty, features


# Ensemble graph for training and testing
def preprocessing(selected_ids, test_count, k, adj_lists, adj_lists_empty, fix):
    #adj_lists, adj_lists_empty, features = load_blog_catalog(selected_ids)

    adj_lists_train = copy.deepcopy(adj_lists_empty)
    adj_lists_test = copy.deepcopy(adj_lists_empty)

    test = [int(x) for x in selected_ids[:test_count]]
    train = [int(x) for x in selected_ids[test_count:len(selected_ids)]]

    selected_ids = set(selected_ids)

    # randomly pick a number within k
    # if not fix:
    #     k = random.randint(1, k)

    for id in train:
        # get list of neighbors
        #neighbors = list(adj_lists[id])
        # delete select id from neighbors
        # neighbors = adj_lists[id].difference(selected_ids)

        neighbors = adj_lists[id]
        sampled_neighbors = set(random.sample(neighbors, k))
        # print(type(sampled_neighbors))
        # sampled_neighbors = set(sampled_neighbors)

        for neighbor in sampled_neighbors:
            if not adj_lists_train[neighbor]:
                adj_lists_train[neighbor] = set()
            adj_lists_train[neighbor].add(id)
        adj_lists_train[id] = sampled_neighbors

        # sampled_neighbors = set()
        # while len(sampled_neighbors) != k:
        #     #rand_ind = np.random.randint(0, len(neighbors))
        #     rand_ind = np.random.choice(len(neighbors))
        #     if neighbors[rand_ind] not in selected_ids:
        #         if not adj_lists_train[neighbors[rand_ind]]:
        #             adj_lists_train[neighbors[rand_ind]] = set()
        #         adj_lists_train[neighbors[rand_ind]].add(id)
        #
        #         sampled_neighbors.add(neighbors[rand_ind])
        # adj_lists_train[id] = sampled_neighbors

    for id in test:
        # get rid of links with other tests data
        neighbors = adj_lists[id].difference(set(test))
        sampled_neighbors = random.sample(neighbors, k)
        # sampled_neighbors = set(sampled_neighbors)

        for neighbor in sampled_neighbors:
            if not adj_lists_test[neighbor]:
                adj_lists_test[neighbor] = set()
            adj_lists_test[neighbor].add(id)
        adj_lists_test[id] = sampled_neighbors


        # neighbors = list(adj_lists[id])
        # sampled_neighbors = set()
        # while len(sampled_neighbors) != k:
        #     #rand_ind = np.random.randint(0, len(neighbors))
        #     rand_ind = np.random.choice(len(neighbors))
        #     if neighbors[rand_ind] not in selected_ids:
        #         # add edges from both direction
        #         if not adj_lists_test[neighbors[rand_ind]]:
        #             adj_lists_test[neighbors[rand_ind]] = set()
        #         adj_lists_test[neighbors[rand_ind]].add(id)
        #         sampled_neighbors.add(neighbors[rand_ind])
        # adj_lists_test[id] = sampled_neighbors

    return adj_lists_train, adj_lists_test, train, test, adj_lists


def run_bc(sample_count, model_name, output):
    # np.random.seed(1)
    #random.seed(1)
    num_nodes = 10312
    feature_dim = 128
    embed_dim = 128
    # load bc data
    selected_id = get_partial_list(1500)


    # Load node2vec features
    adj_lists, adj_lists_empty, features_node2vec = load_blog_catalog(selected_id)

    # Build the graph
    adj_lists_train, adj_lists_test, train, test, adj_lists = preprocessing(selected_id, 300, sample_count, adj_lists,
                                                                                      adj_lists_empty, True)
    print(test)

    # Init to 1
    feat_data = np.ones((num_nodes, feature_dim))
    # for train_id in train:
    #     feat_data[train_id] = features_node2vec[train_id]

    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feature_dim, embed_dim, adj_lists_train, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, embed_dim, adj_lists_train, agg2,
                   base_model=enc1, gcn=False, cuda=False)
    # agg3 = MeanAggregator(lambda nodes: enc2(nodes).t(), cuda=False)
    # enc3 = Encoder(lambda nodes: enc2(nodes).t(), enc2.embed_dim, embed_dim, adj_lists_train, agg3,
    #                base_model=enc2, gcn=False, cuda=False)
    # agg4 = MeanAggregator(lambda nodes: enc3(nodes).t(), cuda=False)
    # enc4 = Encoder(lambda nodes: enc3(nodes).t(), enc3.embed_dim, embed_dim, adj_lists_train, agg4,
    #                base_model=enc3, gcn=False, cuda=False)

    enc1.num_sample = sample_count
    enc2.num_sample = 10
    # enc3.num_sample = 15
    # enc4.num_sample = 20

    graphsage = RegressionGraphSage(enc2)

    feat_data_test = np.ones((10312, 128))
    features_test = nn.Embedding(num_nodes, feature_dim)
    features_test.weight = nn.Parameter(torch.FloatTensor(feat_data_test), requires_grad=False)

    agg1_test = MeanAggregator(features_test, cuda=True)
    enc1_test = Encoder(features_test, feature_dim, embed_dim, adj_lists_test, agg1_test, gcn=False, cuda=False)
    agg2_test = MeanAggregator(lambda nodes: enc1_test(nodes).t(), cuda=False)
    enc2_test = Encoder(lambda nodes: enc1_test(nodes).t(), enc1_test.embed_dim, embed_dim, adj_lists_test, agg2_test,
                   base_model=enc1_test, gcn=False, cuda=False)

    # agg3_test = MeanAggregator(lambda nodes: enc2_test(nodes).t(), cuda=False)
    # enc3_test = Encoder(lambda nodes: enc2_test(nodes).t(), enc2_test.embed_dim, embed_dim, adj_lists_test, agg3_test,
    #                base_model=enc2_test, gcn=False, cuda=False)
    # agg4_test = MeanAggregator(lambda nodes: enc3_test(nodes).t(), cuda=False)
    # enc4_test = Encoder(lambda nodes: enc3_test(nodes).t(), enc3_test.embed_dim, embed_dim, adj_lists_test, agg4_test,
    #                     base_model=enc3_test, gcn=False, cuda=False)
    enc1_test.num_sample = sample_count
    enc2_test.num_sample = 10
    # enc3_test.num_sample = 10
    # enc4_test.num_sample = 10



    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.3)
    times = []
    for epoch in range(1000):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()

        sample_count_epoch = random.randint(1, sample_count)
        adj_lists_train_1, _, _, _, _ = preprocessing(selected_id, 300, sample_count_epoch, adj_lists, adj_lists_empty, False)
        # adj_lists_train_2, _, _, _, _ = preprocessing(selected_id, 300, 10, adj_lists, adj_lists_empty, True)

        enc1.adj_lists = adj_lists_train_1
        enc2.adj_lists = adj_lists_train_1
        enc1.num_sample = sample_count_epoch

        loss = graphsage.loss(batch_nodes, Variable(torch.FloatTensor(features_node2vec[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(epoch, loss)

        if epoch % 10 == 9:
            graphsage_test = RegressionGraphSage(enc2_test)
            graphsage_test.load_state_dict(graphsage.state_dict())
            graphsage_test.eval()

            embed_output = graphsage_test.forward(test)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            print("Cosine similarity: ", cos(embed_output, torch.FloatTensor(features_node2vec[test])).mean(0).item())

    # Save model
    torch.save(graphsage.state_dict(), model_name)
    # run_bc_test_based_on_group(adj_lists_test, feat_data, test, model_name, output, sample_count)

    # embed_output = graphsage.forward(test)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # print("Average Validation Cosine Similarity:", cos(embed_output, torch.FloatTensor(feat_data[test])).mean(0).item())
    # # print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def run_bc_test_based_on_group(adj_lists_test, feat_data, test, model_name, output, edge_count):
    num_nodes = 10312
    feature_dim = 128
    embed_dim = 128

    feat_data_cp = np.ones((10312, 128))
    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data_cp), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feature_dim, embed_dim, adj_lists_test, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, embed_dim, adj_lists_test, agg2,
                   base_model=enc1, gcn=False, cuda=False)
    # agg3 = MeanAggregator(lambda nodes: enc2(nodes).t(), cuda=False)
    # enc3 = Encoder(lambda nodes: enc2(nodes).t(), enc2.embed_dim, embed_dim, adj_lists_test, agg3,
    #                base_model=enc2, gcn=False, cuda=False)
    # agg4 = MeanAggregator(lambda nodes: enc3(nodes).t(), cuda=False)
    # enc4 = Encoder(lambda nodes: enc3(nodes).t(), enc3.embed_dim, embed_dim, adj_lists_test, agg4,
    #                base_model=enc3, gcn=False, cuda=False)

    enc1.num_sample = edge_count
    enc2.num_sample = 10
    # enc3.num_sample = 15
    # enc4.num_sample = 20

    graphsage = RegressionGraphSage(enc2)
    graphsage.load_state_dict(torch.load(model_name))
    graphsage.eval()

    embed_output = graphsage.forward(test)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print("Average Validation Cosine Similarity:", cos(embed_output, torch.FloatTensor(feat_data[test])).mean(0).item())

    #Save Embedding to file
    np.savetxt(output, embed_output.data.numpy())

    # with open("test_id" + str(edge_count) + ".txt", "w") as f:
    #     for item in test:
    #         f.write(str(item) + " ")


def run_bc_test(adj_lists_test, feat_data, test, model_name, output, edge_count):
    num_nodes = 10312
    feature_dim = 128
    embed_dim = 128

    features = nn.Embedding(num_nodes, feature_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feature_dim, embed_dim, adj_lists_test, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, embed_dim, adj_lists_test, agg2,
                   base_model=enc1, gcn=False, cuda=False)
    enc1.num_sample = edge_count
    enc2.num_sample = edge_count
    graphsage = RegressionGraphSage(enc2)

    graphsage.load_state_dict(torch.load(model_name))
    graphsage.eval()


    # test data based on degree (group)
    test_data = []
    with open("../BlogCatalog-data/data_id0.txt", "r") as f:
        vecs = f.readline().split(" ")
        for x in vecs:
            test.append(x)

    embed_output = graphsage.forward(test_data)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print("Average Validation Cosine Similarity:", cos(embed_output, torch.FloatTensor(feat_data[test])).mean(0).item())

    #Save Embedding to file
    np.savetxt(output, embed_output.data.numpy())

    with open("test_id" + str(edge_count) + ".txt", "w") as f:
        for item in test:
            f.write(str(item) + " ")


# Select data from high degree nodes as training & test data
def get_partial_list(count):
    # random.seed(1)
    with open("../BlogCatalog-data/partial_data.txt") as fp:
        candidate_list = [int(x) for x in fp.readline().split(" ")]
        random.shuffle(candidate_list)
        selected_list = candidate_list[:count]
    return selected_list


class RegressionGraphSage(nn.Module):

    def __init__(self, enc):
        super(RegressionGraphSage, self).__init__()
        self.enc = enc
        self.cos_loss = nn.CosineEmbeddingLoss()
        # self.mse_loss = nn.MSELoss()

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds.t()

    def loss(self, nodes, target):
        embeds = self.forward(nodes)
        # return self.mse_loss(embeds, target)
        labels = np.ones(len(target))
        res = self.cos_loss(embeds, target, Variable(torch.FloatTensor(labels)))
        return res


if __name__ == "__main__":
    # Generate new model
    run_bc(10, "GSM_rand_1600.pt", "embedding__rand" + ".txt")


    # for x in range(7, 11):
    #     run_bc(x, "GSM_rand_" + str(x) + ".pt", "embedding" + str(x) + ".txt")

    # Get embedding based on groups
    # selected_id = get_partial_list(1500)
    # adj_lists, adj_list_empty, features = load_blog_catalog(selected_id)
    # _, _, _, _, adj_list = preprocessing(selected_id, 400, 10, adj_lists, adj_list_empty, True)

    # for i in range(10):
    #     feat_data = np.ones((10312, 128))
        # read test data
        # with open("../BlogCatalog-data/data_id" + str(i) + ".txt") as f:
        #     test = [int(x) for x in f.readline().strip().split(" ")]
        # j represents edge number, i represents group id
    # test=[3533, 6242, 9448, 9870, 8384, 9097, 5538, 300, 1433, 9909, 1292, 2733, 3745, 4836, 4731, 4633, 8534, 1439, 9935, 514, 2943, 9793, 2702, 5425, 6643, 7919, 829, 5449, 373, 10270, 7834, 3283, 9354, 9903, 907, 4804, 5865, 6077, 2535, 1614, 3033, 9531, 9163, 1893, 6969, 1906, 2459, 7944, 2033, 7409, 1400, 8224, 4710, 5296, 8623, 1611, 2028, 3701, 2175, 9692, 484, 93, 5404, 8531, 1744, 8826, 3907, 1228, 3509, 4004, 560, 5642, 4613, 903, 3358, 6247, 8510, 4103, 7362, 9845, 4664, 4068, 3932, 1262, 6070, 3365, 2638, 2774, 5135, 2785, 4652, 8156, 2930, 4105, 4489, 7035, 5814, 8616, 2227, 7146, 9013, 3021, 1702, 6162, 4481, 7924, 1553, 4218, 6838, 666, 4455, 7578, 4344, 4726, 6963, 3220, 8586, 7242, 2604, 1521, 6227, 4214, 6972, 9271, 6068, 3231, 5448, 5643, 3352, 550, 4123, 8786, 880, 8840, 1129, 987, 5047, 3473, 1401, 7923, 6337, 5345, 4090, 9285, 4637, 3057, 2631, 9184, 7240, 4531, 5500, 7729, 264, 3239, 2100, 9823, 5401, 8324, 146, 2106, 443, 6681, 8799, 4299, 3800, 6344, 6266, 631, 2750, 9404, 8525, 1217, 4413, 1346, 8532, 875, 8764, 5560, 2601, 6618, 6459, 1002, 1994, 6686, 368, 5008, 3817, 5199, 6111, 5347, 5338, 3022, 4216, 9422, 2232, 3297, 3581, 8154, 496, 7367, 7053, 5369, 6927, 8846, 2737, 4955, 79, 5192, 1463, 8476, 5095, 4784, 5266, 1040, 9488, 1145, 3774, 2163, 2829, 4554, 3966, 8803, 3935, 2547, 468, 644, 7604, 7965, 4907, 6155, 1864, 5319, 2884, 2851, 6437, 8176, 7187, 5137, 8465, 850, 3649, 4033, 838, 7436, 2897, 6267, 8417, 6642, 525, 7594, 6268, 4805, 1530, 5873, 7145, 3507, 1981, 7645, 6946, 964, 6841, 8917, 5677, 2015, 6805, 7761, 1962, 8834, 548, 2319, 7558, 1318, 7470, 10214, 6139, 4850, 8784, 8457, 6219, 5857, 8094, 238, 8091, 9207, 8887, 3386, 431, 5933, 3540, 4226, 8242, 1754, 4445, 7597, 222, 5840, 4875, 6703, 10016, 7122]
        # for j in range(1, 11):
    # run_bc_test_based_on_group(adj_lists, features, test, "GSM_rand_1500_sum.pt", "../BlogCatalog-data/embedding/emd_1500_" + str(11) + ".txt", 3992)

