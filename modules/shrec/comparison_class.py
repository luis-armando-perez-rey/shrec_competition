import numpy as np
import matplotlib.pyplot as plt
import os

class ComparisonClass():
    def __init__(self, query_objects_class, database_objects_class, kl_distance_type = "categorical", alpha = 0.1):
        self.kl_distance_type = kl_distance_type
        self.alpha = alpha
        self.query_class = query_objects_class
        self.database_class = database_objects_class
        self.distances, self.object_distances = self.calculate_distances()
        self.ranking, self.object_ranking = self.rank_objects()
        self.matches, self.object_matches = self.calculate_matches()
        self.precision, self.recall = self.calculate_precision_recall(self.matches)

        self.precision_object, self.recall_object = self.calculate_precision_recall(self.object_matches)

    # def kl_divergence(self, location0, location1, log_sigma0, log_sigma1, pi_cat0, pi_cat1):
    #     epsilon = 1e-7
    #     kl = 0.5 * (np.sum(
    #         np.exp(log_sigma0 - log_sigma1) + np.exp(log_sigma1 - log_sigma0) + (
    #                     np.exp(-log_sigma1) + np.exp(-log_sigma0)) * (location0 - location1) ** 2 - 1, axis=-1))
    #     kl -= 0.1 * np.product(self.database_class.images.shape[1:]) * (
    #                 np.sum(pi_cat0 * (np.log(pi_cat0 + epsilon)-np.log(pi_cat1 + epsilon)), axis=-1) + np.sum(pi_cat1 * (np.log(pi_cat1 + epsilon)-np.log(pi_cat0+ epsilon)),
    #                                                                               axis=-1))
    #     return kl

    def kl_divergence(self,location0, location1, log_sigma0, log_sigma1, pi_cat0, pi_cat1):
        epsilon = 1e-7

        if self.kl_distance_type=="categorical":
                kl = 0.5 * (np.sum(
                np.exp(log_sigma0 - log_sigma1) + np.exp(log_sigma1 - log_sigma0) + (
                        np.exp(-log_sigma1) + np.exp(-log_sigma0)) * (location0 - location1) ** 2 - 2, axis=-1))
                kl -= self.alpha *np.product(self.database_class.images.shape[1:]) * (
                            np.sum(pi_cat0 * np.log(pi_cat1 + epsilon), axis=-1) + np.sum(pi_cat1 * np.log(pi_cat0 + epsilon),
                                                                                  axis=-1))
        elif self.kl_distance_type=="true":
                kl = 0.5 * (np.sum(
                np.exp(log_sigma0 - log_sigma1) + np.exp(log_sigma1 - log_sigma0) + (
                        np.exp(-log_sigma1) + np.exp(-log_sigma0)) * (location0 - location1) ** 2 - 2, axis=-1))
                kl += self.alpha * np.product(self.database_class.images.shape[1:]) * (
                                 np.sum(pi_cat0 * (np.log(pi_cat0 + epsilon)-np.log(pi_cat1 + epsilon)), axis=-1) + np.sum(pi_cat1 * (np.log(pi_cat1 + epsilon)-np.log(pi_cat0+ epsilon)),axis = -1))
        elif self.kl_distance_type=="single":
                kl = 0.5 * (np.sum(
                    np.exp(log_sigma0 - log_sigma1) + (
                    np.exp(-log_sigma1)) * (location0 - location1) ** 2 - 1+log_sigma1-log_sigma0, axis=-1))
                kl += self.alpha * np.product(self.database_class.images.shape[1:]) * (
                                 np.sum(pi_cat0 * (np.log(pi_cat0 + epsilon)-np.log(pi_cat1 + epsilon)), axis=-1))
        elif self.kl_distance_type=="single_categorical":
                kl = 0.5 * (np.sum(
                    np.exp(log_sigma0 - log_sigma1) + (
                    np.exp(-log_sigma1)) * (location0 - location1) ** 2 - 1+log_sigma1-2*log_sigma0, axis=-1))
                kl -= self.alpha * np.product(self.database_class.images.shape[1:]) * (
                    np.sum(pi_cat0 * np.log(pi_cat1 + epsilon), axis=-1))



        return kl


    def calculate_distances(self):
        # Calculate image distances
        distances = np.zeros((self.query_class.num_images, self.database_class.num_images))
        mean_database = self.database_class.mean
        log_z_database = self.database_class.log_var_z
        pi_cat_database = self.database_class.pi_cat
        for query in range(self.query_class.num_images):
            mean_query = self.query_class.mean[query, :]
            log_z_query = self.query_class.log_var_z[query, :]
            pi_cat_query = self.query_class.pi_cat[query, :]
            mean_query = mean_query[np.newaxis, :]
            log_z_query = log_z_query[np.newaxis, :]
            pi_cat_query = pi_cat_query[np.newaxis, :]
            distances[query, :] = self.kl_divergence(mean_query, mean_database, log_z_query, log_z_database, pi_cat_query,
                                               pi_cat_database)

        object_distances = np.zeros((len(self.query_class.mean), len(self.database_class.unique_models)))
        for num_unique_model, unique_model in enumerate(self.database_class.unique_models):
            object_distances[:, num_unique_model] = np.amin(distances[:, self.database_class.models == unique_model], axis=-1)
        return distances, object_distances

    def rank_objects(self):
        ranking = np.argsort(self.distances, axis=-1)
        ranking_objects = np.argsort(self.object_distances, axis = -1)
        return ranking, ranking_objects

    def calculate_matches(self):
        matches = np.zeros((self.distances.shape), dtype=bool)
        object_matches = np.zeros((self.object_distances.shape), dtype=bool)
        for i in range(len(self.query_class.objects_identifiers)):
            matches[i, :] = (self.query_class.objects_identifiers[i] == self.database_class.objects_identifiers[self.ranking[i]])
            object_matches[i, :] = (self.query_class.objects_identifiers[i] == self.database_class.unique_identifiers[self.object_ranking[i]])
        return matches, object_matches

    def create_submission_file(self, path, method_name):
        save_dir = os.path.join(path, method_name)
        os.makedirs(save_dir, exist_ok= True)
        for num_query_model, query_model in enumerate(self.query_class.models):
            print("Saving ranking for photograph {} with model number {}".format(num_query_model, query_model))
            filename = os.path.join(save_dir,str(query_model))
            database_index_rank = self.object_ranking[num_query_model,:]
            query_database_distance = self.object_distances[num_query_model, database_index_rank]
            models_database_ranked = self.database_class.unique_models[database_index_rank]
            with open(filename, 'w+') as f:
                for num_database_model in range(len(query_database_distance)):
                    f.write("{} {}\n".format(query_database_distance[num_database_model], models_database_ranked[num_database_model]))






    def plot_distances(self):
        plt.imshow(np.log10(self.distances + np.unique(self.distances)[1]))
        plt.title(r"$\mathrm{log}_{10}(\mathrm{D}_{\mathrm{KL}})$")
        plt.ylabel("Query object image")
        plt.xlabel("Database object image")
        plt.colorbar()

    def calculate_precision_recall(self, matches):
        total_positives = np.sum(matches)
        tp = 0
        precision = []
        recall = []
        for i in range(matches.shape[1]):
            tp += np.sum(matches[:, i])
            precision.append(tp / ((i + 1) * len(matches)))
            recall.append(tp / total_positives)
        return precision, recall


    def show_ranking_query(self, num_query, max_rank):
        size = int(np.sqrt(self.query_class.images_size))
        object_ranking = self.ranking[num_query,:]
        fig, axes = plt.subplots(self.query_class.num_views, max_rank + 1, figsize = (2.5 * max_rank, self.query_class.num_views * 2.5))
        for view in range(self.query_class.num_views):
            reshaped_query = self.query_class.images_by_view[view][num_query].reshape((size, size))
            ax = axes[view,0]
            ax.imshow(reshaped_query)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("View {}".format(str(view+1)))
            for rank in range(max_rank):
                reshaped_ranked = self.database_class.images_by_view[view][object_ranking[rank]].reshape((size, size))
                ax = axes[view,rank+1]
                ax.imshow(reshaped_ranked, cmap = "gray")
                ax.set_xlabel("Rank {}".format(str(rank + 1)))
                ax.set_xticks([])
                ax.set_yticks([])



