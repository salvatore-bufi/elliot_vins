"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation
from elliot.recommender.latent_factor_models.iALS.iALS_model import iALSModel
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
import os

class iALS(RecMixin, BaseRecommenderModel):
    r"""
    Weighted XXX Matrix Factorization

    For further details, please refer to the `paper <https://archive.siam.org/meetings/sdm06/proceedings/059zhangs2.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        alpha:
        reg: Regularization coefficient

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        WRMF:
          meta:
            save_recs: True
          epochs: 10
          factors: 50
          alpha: 1
          reg: 0.1
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_alpha", "alpha", "alpha", 1, float, None),
            ("_epsilon", "epsilon", "epsilon", 1, float, None),
            ("_reg", "reg", "reg", 0.1, float, None),
            ("_scaling", "scaling", "scaling", "linear", None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train

        self._model = iALSModel(self._factors,
                                self._data,
                                self._nprandom,
                                self._alpha,
                                self._epsilon,
                                self._reg,
                                self._scaling)

    def get_recommendations(self, k: int = 10):
        self._model.prepare_predictions()

        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._data.train_dict.keys()}

    @property
    def name(self):
        return "iALS" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            self._model.train_step()

            print("Iteration Finished")

            self.evaluate(it)

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations()) # cotniene per ogni utente solo le top-K
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                # ogni volta che entro qua sovrascrivo matrice score

                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        self._model.save_weights(self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

