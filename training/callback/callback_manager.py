from training.callback.callback_manager_interface import ICallbackManager
from tensorflow.keras.callbacks import EarlyStopping


class CallbackManager(ICallbackManager):
    def __init__(self):
        self.callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=1,
                verbose=0,
                mode="max",
                restore_best_weights=False,
            )
        ]

    def get_callbacks(self):
        return self.callbacks
