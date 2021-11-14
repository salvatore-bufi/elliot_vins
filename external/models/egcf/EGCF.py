"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from tqdm import tqdm
import numpy as np

from ast import literal_eval as make_tuple

from elliot.dataset.samplers import custom_sampler as cs

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .EGCFModel import EGCFModel


class EGCF(RecMixin, BaseRecommenderModel):
    r"""
    Edge-Based Graph Collaborative Filtering

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        node_edge_factors: Number of shared latent factors for nodes and edges
        weight_size_nodes: Tuple with number of units for each node embedding propagation layer
        weight_size_edges: Tuple with number of units for each edge embedding propagation layer
        weight_size_nodes_edges: Tuple with number of units for each node-edge embedding propagation layer
        batch_size: Batch size
        l_w: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        EGCF:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          node_edge_factors: 128
          weight_size_nodes: (64,)
          weight_size_edges: (64,)
          weight_size_nodes_edges: (64,)
          l_w: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._sampler = cs.Sampler(self._data.i_train_dict)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_node_edge_factors", "node_edge_factors", "node_edge_factors", 128, int, None),
            ("_weight_size_nodes", "weight_size_nodes", "weight_size_nodes", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_weight_size_edges", "weight_size_edges", "weight_size_edges", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_weight_size_nodes_edges", "weight_size_nodes_edges", "weight_size_nodes_edges", "(64,)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_loader", "loader", "loader", 'TextualAttributes', str, None)
        ]
        self.autoset_params()

        self._side_edge_textual = self._data.side_information.TextualAttributes

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        self.edge_index = np.array([row, col])

        list_nodes_edges = []

        for idx in range(self.edge_index.shape[1]):
            list_nodes_edges.append([self.edge_index[0, idx], idx + self._num_users + self._num_items])
            list_nodes_edges.append([self.edge_index[1, idx], idx + self._num_users + self._num_items])

        self.node_edge_index = np.array(list_nodes_edges).transpose()

        list_edges_edges = []

        for e in set(self.node_edge_index[1]):
            nodes_connected_to_e = self.node_edge_index[0, np.argwhere(self.node_edge_index[1] == e)][:, 0].tolist()
            edges_connected_to_e = self.node_edge_index[
                                       1, np.argwhere(self.node_edge_index[0] == nodes_connected_to_e[0])].tolist() + \
                                   self.node_edge_index[
                                       1, np.argwhere(self.node_edge_index[0] == nodes_connected_to_e[1])].tolist()
            edges_connected_to_e = [ee[0] for ee in edges_connected_to_e if ee[0] != e]
            for ee in edges_connected_to_e:
                list_edges_edges.append([e, ee])

        self.edge_edge_index = np.array(list_edges_edges).transpose()
        self.edge_edge_index -= np.min(self.edge_edge_index)

        self._n_layers = len(self._weight_size_nodes)

        self._model = EGCFModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            embed_n_e_k=self._node_edge_factors,
            l_w=self._l_w,
            weight_size_nodes=self._weight_size_nodes,
            weight_size_edges=self._weight_size_edges,
            weight_size_nodes_edges=self._weight_size_nodes_edges,
            n_layers=self._n_layers,
            edge_features=self._side_edge_textual.object.get_all_features(evaluate=True),
            edge_index=self.edge_index,
            node_edge_index=self.node_edge_index,
            edge_edge_index=self.edge_edge_index,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "EGCF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))