from pathlib import Path
from typing import Tuple
import torch
from allennlp.models import load_archive
from allennlp_models.rc import TransformerQAPredictor

from nboost.plugins.qa.base import QAModelPlugin
from nboost.logger import set_logger


class AllenNLPQAModelPlugin(QAModelPlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = set_logger(self.model_dir, verbose=True)
        self.device_id = 0 if torch.cuda.is_available() else -1
        model_path = (Path(self.model_dir) / "model.tar.gz").absolute()

        self.model = TransformerQAPredictor.from_archive(load_archive(str(model_path), self.device_id),
                                                         predictor_name="transformer_qa")

        self.logger.info('Loading AllenNLP from %s' % self.model_dir)

    def get_answer(self, query: str, cvalue: str) -> Tuple[str, int, int, float]:
        result = self.model.predict(query, cvalue)
        span = result['best_span_str']
        score = result['best_span_scores']
        start = cvalue.index(span)
        end = start + len(span)
        return span, start, end, score
