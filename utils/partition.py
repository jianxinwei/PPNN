import numpy as np

def dataset_iid(dataset, num_clients=4):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_clients:
    :return: dict of data index
    """
    num_items = int(len(dataset) / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


