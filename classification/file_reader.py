from gensim.models import KeyedVectors
import copy
import numpy as np
import json


def parse_vector():
    vectors = {}

    with open("vec_all.txt", "r") as f:
        for lines in f:
            line = lines.split(" ")
            if str(line[0]) == "10312":
                pass
            vec = copy.deepcopy(line[1:])
            vectors[line[0]] = np.array([float(x) for x in vec])
    return vectors


def parse_graphsage_vector():
    # read in all test data
    with open("test_id.txt", "r") as f:
        for lines in f:
            ids = [int(x) for x in lines.strip(" ").split(" ")]
    
    count = 0
    #parse partial vector
    vectors = {}
    with open("embed_output1.txt", "r") as f:
        for lines in f:
            line = lines.split(" ")
            vectors[ids[count]] = np.array([float(x) for x in line])
            count += 1
    return ids, vectors


def concat_graphsage_vector(sample_size):
    with open("graphsage_embedding_rand.txt", "w") as fp:
    # with open("graphsage_embedding_" + str(sample_size) + ".txt", "w") as fp:
        for i in range(10):
            # with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/graphsage_embedding/embedding_g_" + str(i) + "_" +  str(sample_size) +".txt", "r") as f:
            with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/graphsage_embedding/gs_rand_emd/emd_test_" + str(i) + ".txt", "r") as f:
                for lines in f:
                    fp.write(lines)


def get_graphsage_vectors(sample_size):
    concat_graphsage_vector(sample_size)
    vectors = {}
    ids = []
    group_range = []
    for i in range(10):
    # with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/data_id2.txt", "r") as f:
        with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/data_id" + str(i) + ".txt", "r") as f:
            vecs = f.readline().strip().split(" ")
            for x in vecs:
                ids.append(int(x))
        group_range.append(len(ids))
        
    # count = 0
    # #with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/src/openne/embedding/embed_output2_10.txt", "r") as fp:
    # with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/graphsage_embedding_" + str(sample_size) + ".txt", "r") as fp:
    #     for lines in fp:
    #         count += 1
    # print(count)

    count = 0
    with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/graphsage_embedding_rand.txt", "r") as fp:
    # with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/graphsage_embedding_" + str(sample_size) + ".txt", "r") as fp:
        for lines in fp:
            line = lines.strip().split(" ")
            vectors[ids[count]] = np.array([float(x) for x in line])
            count += 1
    
    # with open("graphsage_embedding_group.json", "w") as f1:
    #     json.dump(vectors, f1)
    return ids, vectors, group_range

def concate_embed(degree_id_dict):
    # degree_id_dict = degree_id_dict()
    dd = {}
    for i in degree_id_dict:
        for j in degree_id_dict[i]:
            dd[j] = i
    degree_id_dict = dd
    ids, vectors, group_range = get_graphsage_vectors(5)
    groups = {}
    for degree in np.arange(10) + 1:
        _, __v, _ = get_graphsage_vectors(i)
        groups[degree] = __v

    for i in range(3):
        if i == 0:
            for num in np.arange(group_range[i]):
                node_degree = degree_id_dict[ids[num]]
                vectors[ids[num]] = groups[node_degree][ids[num]]
        else:
            for num in np.arange(group_range[i-1], group_range[i]):
                node_degree = degree_id_dict[ids[num]]
                print(len(groups), node_degree)
                vectors[ids[num]] = groups[node_degree][ids[num]]
    return vectors

