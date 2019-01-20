# Reference PyTorch GraphSAGE Implementation
### Author: William L. Hamilton


Basic reference PyTorch implementation of [GraphSAGE](https://github.com/williamleif/GraphSAGE).
This reference implementation is not as fast as the TensorFlow version for large graphs, but the code is easier to read and it performs better (in terms of speed) on small-graph benchmarks.
The code is also intended to be simpler, more extensible, and easier to work with than the TensorFlow version.

Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run the Cora example.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.
There is also a pubmed example (called via the `run_pubmed` function in model.py).

# Documentation for our changes
### graphsage/model.py

#### load_blog_catalog(select_ids): adj_lists, adj_lists_empty, features

This function is used to load blog catalog data into a graph map (adj_lists, a list of dict) and node2vec embedding results (features). The input select_ids is a list of node ids which are selected to be train & test set. It also returns the graph map with all the edges going through those train / test nodes.

#### preprocessing(selected_ids, test_count, k, adj_lists, adj_lists_empty, fix): adj_lists_train, adj_lists_test, train, test, adj_lists

This function splits the selected id list into two lists, a training set (train) and a testing set (test) according to the size of testing group (test_count). Given the graph map adj_lists and adj_lists_empty, we contruct the final graph by sampling k edges for each selected node in the training/testing list and derive adj_lists_train and adj_lists_test.

#### get_partial_list(count): selected_list

This function reads the id list from a file and selects a sub-list from it as the selected training & testing set.

#### run_bc(sample_count, model_name, output): 

This function runs the whole process of training a model based on the blog catalog data. It first selects the training & testing set and then load the node2vec data and the graph map according to this. Next it initializes the whole model and loads corresponding data. Then it starts run a number of epochs, calculate the loss and propagate it back. The training process also outputs the cosine similarity on the testing model and data every 10 epochs. In the end, it saves the parameters into a file with file name model_name.

#### run_bc_test_based_on_group(adj_lists_test, feat_data, test, model_name, output, edge_count):

This function is different from the training process in that it takes the test graph structure and feature data as input already. The testing data should be split into several groups according to some rules such as node degrees. The output result should be corresponding embedding for each node, which will be saved in model_name.

#### run_bc_test(adj_lists_test, feat_data, test, model_name, output, edge_count):

This function is actually very similar to run_bc_test_based_on_group. This intends to run tests on the whole testing data instead of separate groups.

### graphsage/encoders.py

We change the activation function of the last step in the encoder structure to tanh function instead of ReLU so that it can suit the regression task better. Besides we also try to add one more fully connect layer to encode the embedding in order to encode more information but fail to improve the performance. This part is now commented out in the forward().

### graphsage/aggregators.py

Currently there is no change over here. One key point is that it is using the mean aggregator function and it will sum up all sampled neighbours and divide it with the count of neighbour vectors used instead of just the general sample size because some nodes may lack enough neighbours for sampling. 