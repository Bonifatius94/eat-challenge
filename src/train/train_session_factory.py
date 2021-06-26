
from .session import NaiveTrainingSession


class TrainSessionFactory:

    def create_train_session(self, train_type: str, params: dict):

        if train_type == 'naive': return NaiveTrainingSession(params)
        # TODO: add more training sessions here ...
