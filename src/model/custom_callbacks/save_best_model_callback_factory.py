import platform
from . import SaveBestAccuracyCallback

class SaveBestModelCallbackFactory:

    def get_callback(self, ckpt_path: str):

        if platform.system() == 'Windows':
            return SaveBestAccuracyCallback(ckpt_path)
        elif platform.system() == 'Linux':
            return ModelCheckpoint(
                filepath=ckpt_path, save_weights_only=False,
                monitor='val_accuracy', mode='max', save_best_only=True)
        else:
            raise NotImplementedError('Mac is not supported')
