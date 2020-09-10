from typing import Tuple
import time
from nboost.plugins import Plugin
from nboost.delegates import ResponseDelegate
from nboost.database import DatabaseRow
from nboost import defaults
from nboost.logger import set_logger


class QAModelPlugin(Plugin):
    def __init__(self,
                 max_query_length: type(defaults.max_query_length) = defaults.max_query_length,
                 model_dir: str = defaults.qa_model_dir,
                 max_seq_len: int = defaults.max_seq_len,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.max_query_length = max_query_length
        self.max_seq_len = max_seq_len
        self.logger = set_logger('qamodel', verbose=True)

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow):
        if response.cvalues:
            start_time = time.perf_counter()
            responses = []
            for idx, cvalue in enumerate(response.cvalues):
                answer, start_pos, stop_pos, score = self.get_answer(response.request.query, cvalue)

                self.logger.info(f"{response.request.qa_threshold} \t {answer}, {start_pos}, {stop_pos}, {score}")

                responses.append({
                    'answer_text': answer,
                    'answer_start_pos': start_pos,
                    'answer_stop_pos': stop_pos,
                    'answer_score': score,
                })

            db_row.qa_time = time.perf_counter() - start_time

            response.set_path(f'body.nboost.qa', responses)

    def get_answer(self, query: str, cvalue: str) -> Tuple[str, int, int, float]:
        """Return answer, start_pos, end_pos, score"""
        raise NotImplementedError()
