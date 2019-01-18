from collections import OrderedDict
import json

def get_node_degree():
    node_degree = {}
    # calculate node degree
    with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/src/openne/edgelist.txt", "r") as f:
        for line in f:
            nodes = line.split(" ")

            if nodes[0] not in node_degree:
                node_degree[nodes[0]] = 1
            else:
                node_degree[nodes[0]] += 1
        # for key in node_degree:
        #     if node_degree[key] == 3992:
        #         print (key)
        #print(sorted(node_degree.items(), key=lambda kv: kv[1]))
        return node_degree

# get the representation of dictionary that is sorted by its degree
# sorted_by_value = sorted(node_degree.items(), key=lambda kv: kv[1])

#Only consider nodes with degree less than 500
# Rest of the nodes has a higher range
def get_node_degree_dp_distribution():
    # node_degree = get_node_degree()
    node_degree = {}
    # calculate node degree
    with open("/Users/StephanieYuan/Desktop/Capstone/OpenNE/src/openne/edgelist.txt", "r") as f:
        for line in f:
            nodes = line.split(" ")

            if nodes[0] not in node_degree:
                node_degree[nodes[0]] = 1
            else:
                node_degree[nodes[0]] += 1
        # for key in node_degree:
        #     if node_degree[key] == 3992:
        #         print (key)
        #print(sorted(node_degree.items(), key=lambda kv: kv[1]))
    sorted_node_degree = OrderedDict(sorted(node_degree.items(), key=lambda x: x[1]))
    #key: id of the group, value: nodes whose degree are within range of the 
    group_dict = {}
    group_id = 1
    group_dict[group_id] = []
    
    # for key, val in sorted_node_degree.items():
    #     if val > (group_id - 1) * 5:
    #         #increment group_id, create new entry
    #         group_id += 1
    #         group_dict[group_id] = []
    #     else:
    #         group_dict[group_id].append(int(key))

    # print (group_dict)
    
    max_id = 0

    for key, val in sorted_node_degree.items():
        
        if val > 500:
            if max_id not in group_dict:
                group_dict[max_id] = []
            group_dict[max_id].append(key)
        if val > group_id  * 5:
            group_id += 1
            max_id = group_id + 1
            group_dict[group_id] = []
        group_dict[group_id].append(key)
    

    # dump to json format
    # with open("node_degree_distribution.json", "w") as jsondata :
        # json.dump(group_dict, jsondata)

    return group_dict


def sort_data_by_degree():
    degree_group_dict = get_node_degree_dp_distribution()
    X, Y = read_label("/Users/StephanieYuan/Desktop/Capstone/OpenNE/data/blogCatalog/bc_labels.txt")
    X_sorted = []
    Y_sorted = []

    for _, li in degree_group_dict.items():
        for val in li:
            X_sorted.append(val)
            ind = X.index(val)
            Y_sorted.append(Y[ind])
    return X_sorted, Y_sorted


def read_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def degree_id_dict():
    node_degree_dict = get_node_degree()
    degree_dict = sorted(node_degree_dict.items(), key=lambda kv: kv[1])

    degree_id_dict = {}

    for i in range(1, 11):
        X_all = []
        for x, y in degree_dict:
            if y == i:
                X_all.append(int(x))
        degree_id_dict[i] = X_all
    return degree_id_dict
