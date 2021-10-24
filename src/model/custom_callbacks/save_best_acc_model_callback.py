import tensorflow as tf


class SaveBestAccuracyCallback(tf.keras.callbacks.Callback):

    def __init__(self, cpt_path: str):#, model_to_save: tf.keras.Model):
        super(SaveBestAccuracyCallback, self).__init__()
        self.best_accuracy = 0
        # self.model_to_save = model_to_save
        self.cpt_path = cpt_path

    def on_test_end(self, logs=None):
        if 'accuracy' not in logs:
            tf.print("No accuracy in logs, aborting...")
            return

        acc = float(logs['accuracy'])
        if acc > self.best_accuracy:
            tf.print("New best Model, saving!")
            self.best_accuracy = acc
            self.model.save_weights(self.cpt_path, True)
