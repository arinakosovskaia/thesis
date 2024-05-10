import torch.nn as nn
import transformers
from transformers import XLMRobertaModel, XLMRobertaConfig


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = XLMRobertaModel.from_pretrained(model_name)
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            setattr(model, "roberta", shared_encoder)
            model.classifier.out_proj = nn.Linear(1024, 2)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


model_name = "sismetanin/xlm_roberta_large-ru-sentiment-rusentiment"
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "ethics": transformers.AutoModelForSequenceClassification,
        "toxic": transformers.AutoModelForSequenceClassification,
        "appropriate": transformers.AutoModelForSequenceClassification
    },
    model_config_dict={
        "ethics": transformers.AutoConfig.from_pretrained(model_name),
        "toxic": transformers.AutoConfig.from_pretrained(model_name),
        "appropriate": transformers.AutoConfig.from_pretrained(model_name)
    },
)

