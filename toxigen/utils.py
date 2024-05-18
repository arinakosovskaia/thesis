from toxigen.language_models import GPT3, ALICE, GPT2
import pandas as pd

def alice(prompt, classifier, endpoint_url=None, api_key=None, mode="neutral", num_to_generate=1, device="cpu", model="GPT3"):
    if model == 'GPT3':
        if endpoint_url is None or api_key is None:
            return "Empty api_key or endpoint_url"
        language_model = ALICE(GPT3(endpoint_url=endpoint_url, apikey=api_key), classifier, mode, device="cpu")
    else:
        language_model = ALICE(GPT2(), classifier, mode, device="cpu")
    
    return [language_model(prompt) for _ in range(num_to_generate)]

def label_annotations(annotated):
    # Annotations should be the annotated dataset
    label = ((annotated.toxicity_ai + annotated.toxicity_human) > 5.5).astype(int)
    labeled_annotations = pd.DataFrame()
    labeled_annotations["text"] = [i for i in annotated.text.tolist()]
    labeled_annotations["label"] = label
    return labeled_annotations