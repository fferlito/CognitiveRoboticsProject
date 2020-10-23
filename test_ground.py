import numpy as np
import os
import os.path as osp

#import src.utils.visualization as visl

from sklearn.cluster import KMeans

#from src.data.dataloader import DataLoader
#from src.online_kmean import OnlineKMeans
#from src.utils.logging import setup_logger


def compute_purity(labels, clusters, num_clusters):
    """
    Compute the purity of clusters with respect to labels
    :param labels: The ground truth labels
    :param clusters: K-Mean clusters
    :param num_clusters: The number of clusters in "clusters" (this function input)
    :return:
    """
    # For each cluster, find the ground truth cluster with the largest overlap
    matched_labels = []
    purity_list = []
    for idx in range(num_clusters):
        gt = labels[np.nonzero(clusters == idx)]
       # print('IDX: ' + str(idx))
        unique_gt, unique_counts = np.unique(gt, return_counts=True)


        if(len(unique_counts) > 0):
            max_count_idx = np.argmax(unique_counts)
            best_label = unique_gt[max_count_idx]
            #print(max_count_idx)
            #print(best_label)

            num_points = len(gt)
            num_points_in_cluster = unique_counts[max_count_idx]

            purity_list.append(num_points_in_cluster / num_points)
            matched_labels.append(best_label)
        else:
            purity_list.append(0)

    #for idx in range(num_clusters):
        #print("Cluster {} matched to label {}. Its purity is {:.4f}".format(idx, matched_labels[idx], purity_list[idx]))
    #print("Average purity is {:.4f}".format(np.mean(purity_list)))
    return np.mean(purity_list)


def purity_analysis(cfg):
    """
    Analyze overparameterized kmean model's purity with respect to the label ground truth.
    :return:
    """
    # Set up logger
    task_name = "purity_analysis"
    output_dir = osp.join(cfg.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                          timestamp=True)

    logger.info(f"Running with configuration: \n{cfg}")

    # Load dataset
    num_features = 100
    data_loader = DataLoader(data_dir=cfg.data.path, dataset=cfg.data.name, split=cfg.data.split,
                             num_features=num_features)

    data = data_loader.get_data()
    labels = data_loader.get_labels()
    num_samples, num_features = data.shape

    # Run the batch kmeans
    ref_model = KMeans(n_clusters=data_loader.num_clusters)
    batch_labels = ref_model.fit_predict(data)

    # Run the online kmeans
    if cfg.data.name in ['mnist', 'fashion_mnist']:
        num_clusters = 12
    elif cfg.data.name == 'news_groups':
        num_clusters = 32
    else:
        raise NotImplementedError

    model = OnlineKMeans(num_features=num_features, num_clusters=num_clusters)
    clusters = model.fit_predict(data)

    visl.show_center_and_data(ref_model.cluster_centers_, model.centroid,
                              title=f'Online K-means centroids (Blue dot) and \nOffline K-means centroids (Red '
                                    f'cross) on the Dataset {cfg.data.name}')

    logger.info("Purity with respect to labels")
    compute_purity(labels, clusters, num_clusters, logger)

    logger.info("Purity with respect to batck kmeans")
    compute_purity(batch_labels, clusters, num_clusters, logger)


def binary_search_for_smallest_clusters_size(data, target_cost, min_size, max_size, num_features, **kwargs):
    """
    Find the smallest number of clusters for the online k-means so that its k-means cost is lower than the
    target_cost.

    We are assuming that overparametrized k-means cost is monotonically increased w.r.t the cluster size.
    Using binary search.
    :param data:
    :param target_cost:
    :param min_size: The minimum number of clusters required.
    :param max_size: The maximum number of clusters needed.
    :param num_features:
    :param kwargs: key word parameters for the OnlineKMeans
    :return:
    """
    left = min_size
    right = max_size

    print(f'Start searching...')

    min_clusters = -1
    while left <= right:
        middle = (left + right) // 2

        print(f"{' ':*>3}Trying cluster size {middle} ...")

        model = OnlineKMeans(num_features=num_features, num_clusters=middle, **kwargs)
        model.fit(data)
        curr_cost = model.calculate_cost(data)

        print(f"{' ':*>3}Cost {curr_cost}")

        if np.isclose(curr_cost, target_cost):
            print(f"{' ':*>3}Find the best cost {curr_cost} with cluster {middle}")
            min_clusters = middle
            break
        elif curr_cost < target_cost:
            min_clusters = middle
            right = middle - 1
        else:
            left = middle + 1

    print(f"{' ':*>3}Find the min clusters {min_clusters}")

    return min_clusters


def find_the_minimum_cluster_to_reach_cost(cfg, base_num_clusters=None, num_run=10, logger=None):
    """
    Find the minimum number of cluster needed to reach the same k-means cost produced by the KMean++ algorithm
    :param cfg: configuration
    :param base_num_clusters: The basic number of clusters in the batch k-means. If None, then we will use the number
    of classes in the data set.
    :param num_run: The number of run for the online k-means
    :param logger: If logger is provided, we will use that logger.
    :return: The average min # of clusters
    """

    # Set up logger
    if logger is None:
        task_name = "find_min_cluster"
        output_dir = osp.join(cfg.output_dir, task_name)
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                              timestamp=True)

        logger.info(f"Running with configuration: \n{cfg}")

    # Load the data
    num_features = 100
    data_loader = DataLoader(data_dir=cfg.data.path, dataset=cfg.data.name, split=cfg.data.split,
                             num_features=num_features)
    data = data_loader.get_data()

    # Find the cost of KMean++
    num_clusters = data_loader.num_clusters if base_num_clusters is None else base_num_clusters
    ref_model = KMeans(n_clusters=num_clusters)
    ref_model.fit(data)
    target_cost = ref_model.inertia_

    logger.info(f"target cost is {target_cost}")

    # Find the minimum number of clusters needed so that the online kmean cost is lower than the target cost
    left = num_clusters
    # right = np.square(left)  # Only consider up to k^2 clusters (because that is enough)
    # Since the relative ratio of overparametrization is around 1.5, we just manually set the maximum number of
    # clusters to search as 4 (this is a ad-hoc solution)
    right = 4 * num_clusters

    min_cluster_list = []
    for _ in range(num_run):
        # Shuffle the data
        num_samples = data.shape[0]
        shuffle_idx = np.random.permutation(num_samples)
        shuffled_data = data[shuffle_idx]
        c = binary_search_for_smallest_clusters_size(shuffled_data, target_cost, left, right,
                                                     data_loader.num_features)

        min_cluster_list.append(c)

    avg_min = np.mean(min_cluster_list)

    logger.info(f'Total result {min_cluster_list}')
    logger.info(f'Average min clusters {avg_min}')
    logger.info(f'Relative ratio is {avg_min / num_clusters}')

    return avg_min


def find_min_with_various_base(cfg):
    """ Find the minimum number of cluster with various base """
    task_name = "find_min_various_base"
    output_dir = osp.join(cfg.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                          timestamp=True)

    logger.info(f"Running with configuration: \n{cfg}")

    base_list = np.linspace(0, 100, num=11).astype(np.int)
    base_list[0] = 1

    avg_min_list = []
    ratio_list = []
    for base in base_list:
        logger.info(f"Running with base={base}")
        avg_min = find_the_minimum_cluster_to_reach_cost(cfg, base_num_clusters=base, num_run=10, logger=logger)
        avg_min_list.append(avg_min)
        ratio_list.append(avg_min / base)

    # Visualize the result
    fig, ax = visl.show_relative_ratio_with_base(base_list, ratio_list, title=f"Data set: {cfg.data.name}")


def search_learning_rate(cfg):
    """ Search the learning rate of online k-means """
    task_name = "search_learning_rate"
    output_dir = osp.join(cfg.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                          timestamp=True)

    logger.info(f"Running with configuration: \n{cfg}")

    # Load the data
    num_features = 100
    data_loader = DataLoader(data_dir=cfg.data.path, dataset=cfg.data.name, split=cfg.data.split,
                             num_features=num_features)
    data = data_loader.get_data()

    # Find the cost of KMean++
    num_clusters = data_loader.num_clusters
    ref_model = KMeans(n_clusters=num_clusters)
    ref_model.fit(data)
    target_cost = ref_model.inertia_

    logger.info(f"target cost is {target_cost}")

    # Find the minimum number of clusters needed so that the online kmean cost is lower than the target cost
    left = num_clusters
    right = np.square(left) * 6  # 6 for news group and 2 for the MNIST dataset

    # Shuffle the data
    num_samples = data.shape[0]
    shuffle_idx = np.random.permutation(num_samples)
    shuffled_data = data[shuffle_idx]

    # Fix t0
    t0 = 0
    c_prime_list = np.linspace(1, 100, num=11)
    logger.info(f'Fixed t0 to {t0}, Trying c_prime: {c_prime_list}')

    min_cluster_list = []
    for c_prime in c_prime_list:
        lr = (c_prime, t0)
        c = binary_search_for_smallest_clusters_size(shuffled_data, target_cost, left, right,
                                                     data_loader.num_features, lr=lr)
        min_cluster_list.append(c)

    avg_min = np.mean(min_cluster_list)

    logger.info(f'Total result {min_cluster_list}')
    logger.info(f'Average min clusters {avg_min}')
    logger.info(f'Relative ratio is {avg_min / num_clusters}')

    # Fix c_prime
    c_prime = 10
    t0_list = np.linspace(1, 100, num=11)
    logger.info(f'Fixed c_prime to {c_prime}, Trying t0: {t0_list}')

    min_cluster_list = []
    for t0 in t0_list:
        lr = (c_prime, t0)
        c = binary_search_for_smallest_clusters_size(shuffled_data, target_cost, left, right,
                                                     data_loader.num_features, lr=lr)
        min_cluster_list.append(c)

    avg_min = np.mean(min_cluster_list)

    logger.info(f'Total result {min_cluster_list}')
    logger.info(f'Average min clusters {avg_min}')
    logger.info(f'Relative ratio is {avg_min / num_clusters}')


def main():
    import src.config.mnist
    import src.config.fashion_mnist
    import src.config.news_groups

    packages = [src.config.mnist, src.config.fashion_mnist, src.config.news_groups]
    cfg_list = [p.get_cfg_defaults() for p in packages]

    for cfg in cfg_list:
        # find_the_minimum_cluster_to_reach_cost(cfg)
        # find_min_with_various_base(cfg)
        # search_learning_rate(cfg)
        purity_analysis(cfg)


if __name__ == '__main__':
    main()
