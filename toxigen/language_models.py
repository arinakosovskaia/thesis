import requests
import json
from toxigen.alice import beam_search
from toxigen.alice_general import beam_search_general
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np

class GPT3(object):
    def __init__(self, endpoint_url, apikey):
        self.apikey = apikey
        self.endpoint_url = endpoint_url
        self.name = 'GPT3'

    def __call__(self, prompt, topk=1, max_tokens=1):
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = [p.replace("'", "").replace('"', "") for p in prompt]
        payload = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.5,
            "n": 1,
            "stream": False,
            "logprobs": topk,
            "stop": ["<|endoftext|>", "\\n", '\n']
        }
        r = requests.post(self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.apikey}",
                "Content-Type": "application/json"
            },
            json = payload
        )       
        output = json.loads(r.content)
        return output

    def from_prompt(self, prompt, topk=10, max_tokens=10):
        output = self.__call__(prompt, topk, max_tokens)
        return output['response']["choices"][0]["text"]

class GPT2(object):
    def __init__(self, model_path='ai-forever/rugpt3large_based_on_gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.name = "GPT2"
        
    def probabilities(self, text, topk=5):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs,
                  max_new_tokens=1,
                  num_beams=topk,
                  num_return_sequences=topk,
                  return_dict_in_generate=True,
                  output_scores=True,
                  )
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
        input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:].to("cpu")
        tokens = [self.tokenizer.decode(tok) for tok in generated_tokens]
        scores = [score.numpy()[0] for score in transition_scores.to("cpu")]

        result = dict()
        for tok, score in zip(generated_tokens, transition_scores):
          token = self.tokenizer.decode(tok)
          score = score.to("cpu").numpy()
          result[token] = np.array(score)
        return tokens, scores, result
        
    def __call__(self, input_text, topk=1, max_tokens=1):
        self.model.eval()
        tokens_array, scores_array, concat_array = [], [], []
        for text in input_text:
          tokens, scores, concatenated = self.probabilities(text, topk)
          tokens_array.append([tokens])
          scores_array.append([scores])
          concat_array.append(concatenated)
        return tokens_array, scores_array, concat_array
    
class ALICE(object):
    def __init__(self, language_model, classifier, mode, device="cpu"):
        self.classifier = classifier
        self.language_model = language_model
        self.device = device
        self.mode = mode

    def __call__(self, prompt):
        return self.generate(prompt)

    def generate(self, prompt):
        if self.mode == "neutral":
            flag = 0
        else:
            flag = 1
        if self.language_model.name == 'GPT3':
            return beam_search(prompt, self.language_model, self.classifier, flag, self.device) 
        else:
            return beam_search_general(prompt, self.language_model, self.classifier, flag, self.device) 
        
