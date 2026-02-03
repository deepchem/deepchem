
# (imports stay exactly as DeepChem has them)
# I am showing ONLY the relevant class part here

class HuggingFaceModel(TorchModel):

    def predict(
        self,
        dataset,
        transformers=None,
        **kwargs
    ):
        if self.task == "generation":
            return self._predict_generation(dataset, **kwargs)
        return super().predict(dataset, transformers=transformers)
