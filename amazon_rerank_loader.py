import gzip
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import pickle 
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from texttable import Texttable
import gensim
import copy
from collections import Counter, defaultdict


b_type_dict = {
    "also_viewed": 0,
    "also_bought": 0,
    "bought_together": 0,
    "buy_after_viewing": 0
}

edge_dict = {
    "[0, 0, 0, 0]": 0,
    "[1, 0, 0, 0]": 0,
    "[0, 1, 0, 0]": 0,
    "[0, 0, 1, 0]": 0,
    "[0, 0, 0, 1]": 0,
    "[1, 1, 0, 0]": 0,
    "[1, 0, 1, 0]": 0,
    "[1, 0, 0, 1]": 0,
    "[0, 1, 1, 0]": 0,
    "[0, 1, 0, 1]": 0,
    "[0, 0, 1, 1]": 0,
    "[1, 0, 1, 1]": 0,
    "[0, 1, 1, 1]": 0,
    "[1, 0, 1, 1]": 0,
    "[1, 1, 0, 1]": 0,
    "[1, 1, 1, 0]": 0,
    "[1, 1, 1, 1]": 0
}


_urls = "---"
# cat= "Electronics"
cat = "Video_Games"
# cat = "Clothing_Shoes_and_Jewelry"
# cat = "Musical_Instruments"
# cat = "Movies_and_TV"
num_node_feat = 300
num_of_min_interactions = 30


class AmazonDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(AmazonDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # self.data = torch.load(self.processed_paths[0])

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["meta_" + cat + ".json.gz", "train_ratings_" + cat + ".txt", "test_ratings_" + cat + ".txt"]

    @property
    def processed_file_names(self):
        return "Amazon_%s.pt" % cat

    def download(self):
        raise NotImplementedError("please download and unzip dataset from %s, and put it at %s" 
            % (_urls + "meta_" + cat + ".json.gz", self.raw_dir))

    def parse(self, path):
      g = gzip.open(path, "rb")
      for l in g:
        yield eval(l)

    def getDF(self, path):
      i = 0
      df = {}
      print("Reading data...")
      for d in tqdm(self.parse(path)):
        df[i] = d
        i += 1
      return pd.DataFrame.from_dict(df, orient="index")

    def amazon_nodes(self, g):
        feat, is_user = [], []
        for n, d in g.nodes(data=True):
            feat.append((n, d["node_feat"]))
            is_user.append(d["node_type"] == "user")
        feat.sort(key=lambda item: item[0])
        
        return torch.FloatTensor([item[1] for item in feat]), torch.ByteTensor(is_user)

    def get_inner_id(self, g, users_in_train, users_in_test):
        users_inner_train, users_inner_test = [], []
        revert_graph = nx.Graph()
        inner_id_dict = {}
        for n, d in g.nodes(data=True):
            revert_graph.add_node(d["old_label"], inner_id=n)
            inner_id_dict[n] = d["old_label"]
            

        for user, iteracted_i in users_in_train.items():
            for item, rating, prediction in iteracted_i:
                if revert_graph.has_node(user) and revert_graph.has_node(item):
                    users_inner_train.append([revert_graph.node[
                        item]["inner_id"], revert_graph.node[user]["inner_id"], float(rating), float(prediction)])

        for user, iteracted_i in users_in_test.items():
            for item, rating, prediction in iteracted_i:
                if revert_graph.has_node(user) and revert_graph.has_node(item):
                    users_inner_test.append([revert_graph.node[
                        item]["inner_id"], revert_graph.node[user]["inner_id"], float(rating), float(prediction)])

        pickle.dump(inner_id_dict, open(cat.split("_")[0] + "_inner_id.pkl", "wb"))


        return torch.LongTensor(users_inner_train), torch.FloatTensor(users_inner_test)


    def amazon_edges(self, g):
        e = {}
        y = {}
        count_edge = np.array([0, 0, 0, 0])
        for n1, n2, d in g.edges(data=True):
            
            # item-item edges
            if "b_type" in d:
                e_t = [int(x in d["b_type"]) for x in sorted(list(b_type_dict.keys()))]
                e[(n1, n2)] = e_t
                count_edge = count_edge + np.array(e_t)
                y[(n1, n2)] = -1
            # training
            elif "rating" in d:
                e[(n1, n2)] = d["prediction"]
                y[(n1, n2)] = d["rating"][0]
            #test
            else:
                e[(n1, n2)] = d["prediction"]
                y[(n1, n2)] = -1


        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        y = torch.LongTensor(list(y.values()))


        return edge_index, edge_attr, y, count_edge

    def process_user_item_iteractions(self, node_idx_dict):
        users_in_train = defaultdict(list)
        users = set()
        with open(self.raw_paths[1], "r", encoding="utf-8") as f:   # ratings train
            for line in f:
                u, i, r, y = line.strip().split()
                if i in node_idx_dict:
                    users_in_train[u].append((i, r, y))
                    users.add(u)

        users_in_test = defaultdict(list)
        with open(self.raw_paths[2], "r", encoding="utf-8") as f:
            for line in f:
                u, i, r, y = line.strip().split()
                if i in node_idx_dict:
                    users_in_test[u].append((i, r, y))
                    users.add(u)

        dup_users_in_train = copy.deepcopy(users_in_train)
        dup_users_in_test = copy.deepcopy(users_in_test)

        for k, v in dup_users_in_train.items():
            if len(v) < num_of_min_interactions:
                del users_in_train[k]


        for k, v in dup_users_in_test.items():
            if (len(v) < num_of_min_interactions) or (k not in users_in_train):
                del users_in_test[k]

        n_u, n_v, n_r = self.print_statistics(users_in_train, users_in_test)
        return users_in_train, users_in_test, users, n_u, n_v, n_r

    def print_statistics(self, train_data, test_data):
        items = set()
        users = set()
        ratings = 0
        for u, v in train_data.items():
            users.add(u)
            for i in v:
                items.add(i[0])
                ratings += 1
        for u, v in test_data.items():
            users.add(u)
            for i in v:
                items.add(i[0])
                ratings += 1
        return len(users), len(items), ratings



    def process(self):
        data_list = []
        data = self.getDF(self.raw_paths[0])
        self.num_item = len(data.index)
        # node_idx_dict = self.getDict(data.asin)
        d2v_model = gensim.models.doc2vec.Doc2Vec.load(self.raw_dir + "/reviews_" + cat + ".d2v")

        nodes = set(data.asin) & set(d2v_model.docvecs.doctags.keys())


        graph = nx.DiGraph()
        cnt = 0 

        print("Constructing graph...")

        for idx, item in tqdm(data.iterrows(), total=data.shape[0]):


            # add nodes
            if item["asin"] in nodes:
                graph.add_node(item["asin"], node_feat=d2v_model.docvecs[item["asin"]], asin=item["asin"], node_type="item")

                # add edges
                if not pd.isna(item["related"]):
                    relations = item["related"]
                    for b_type in relations.keys():
                        b_type_dict[b_type] += len(relations[b_type])
                        for dest in relations[b_type]:
                            if dest in nodes:
                                if (item["asin"], dest) in graph.edges:
                                    graph.edges[item["asin"], dest]["b_type"].append(b_type)
                                else:
                                    graph.add_edge(item["asin"], dest, b_type=[b_type])


        # add user-item, score information
        # copy and add new nodes
        users_in_train, users_in_test, users,  n_u, n_v, n_r = self.process_user_item_iteractions(nodes)
        for user, iteracted_i in users_in_train.items():
            cnt = 0
            graph.add_node(user, node_feat=np.zeros(num_node_feat), node_type="user")
            for item, rating, prediction in iteracted_i:     
                if graph.has_node(item):
                    cnt += 1
                    graph.add_edge(item, user, prediction=[float(prediction), 0, 0, 0], rating=[int(rating), 0, 0, 0])
                    graph.node[user]["node_feat"] = (graph.node[user]["node_feat"] + d2v_model.docvecs[item]) / cnt


            for item, rating, prediction in users_in_test[user]:
                if graph.has_node(item):
                    cnt += 1
                    graph.add_edge(item, user, prediction=[float(prediction), 0, 0, 0])
                    graph.node[user]["node_feat"] = (graph.node[user]["node_feat"] + d2v_model.docvecs[item]) / cnt


        # remove all isolates
        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)
        graph = nx.convert_node_labels_to_integers(graph, label_attribute="old_label")

        tr, ts = self.get_inner_id(graph, users_in_train, users_in_test)

        node_attr, is_user = self.amazon_nodes(graph)
        edge_index, edge_attr, y, count_edge = self.amazon_edges(graph)

        print("Graph completed!")

        amazon_data = Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_data=tr,
        test_data=ts,
        is_user=is_user,
        y=y
        )



        t = Texttable()
        t.add_rows([["cat", "#users", "#items", "#ratings", "sparsity", "#also_bought", "#also_viewed", 
            "#bought_together", "#buy_after_viewing"], 
            [cat, n_u, graph.number_of_nodes()-n_u, n_r, float(n_r)/(n_u * n_v),
            # graph.number_of_edges(), 
            count_edge[0],
            count_edge[1],
            count_edge[2],
            count_edge[3]]])
        print(t.draw())

        data_list = [amazon_data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




if __name__ == "__main__":
    root = "./data/Amazon"
    dataset = AmazonDataset(root=root)
    dataset.process()
    




