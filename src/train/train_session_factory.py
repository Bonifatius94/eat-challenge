
from .session import NaiveTrainingSession, NoisyTrainingSession


class TrainSessionFactory:

    def create_train_session(self, train_type: str, params: dict):

        if train_type == 'naive': return NaiveTrainingSession(params)
        if train_type == 'noisy': return NoisyTrainingSession(params)
        # TODO: add more training sessions here ...
