import mlflow.pyfunc

class ModelLoader:
    _model = None

    @classmethod
    def load_model(cls, uri: str = "models:/my_model/develop"):
        pass
        """Load and cache the model once."""
        # if cls._model is None:
        #     logger.info(f"Loading model from {uri} ...")
        #     try:
        #         model = mlflow.pyfunc.load_model(uri)

        #         # If it's a PyTorch model flavor, switch to eval mode
        #         if hasattr(model, "_model_impl") and isinstance(model._model_impl, torch.nn.Module):
        #             model._model_impl.eval()
        #             logger.info("Model set to eval mode (PyTorch).")

        #         cls._model = model
        #         logger.info("Model loaded successfully.")
        #     except Exception as e:
        #         logger.exception(f"Failed to load model from {uri}: {e}")
        #         raise e
            

        # return cls._model
        return
