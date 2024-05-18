import torch
import itertools
import copy
import numpy as np
from nltk.corpus import stopwords

def omit_(token, score, list1,list2):
    if token not in list1 and token.replace(' ','') not in list1 and token in list2:
       return -100
    return score

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

def beam_search_general(prompt,
                language_model,
                classifier,
                mode, # if 1, Toxic, if 0 Neutral
                device,
                end_token="\n",
                weights=[.5, .5],
                num_beams=10,
                vocab_size=100,
                max_length=30,
                length_penalty=1):
    """Generate sequences for each example with beam search."""
    vocab_size = min(5, vocab_size) 
    pad_token_id = '<|pad|>'
    eos_token_ids = [end_token]
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(1)
    ]
    stops = list(stopwords.words('russian'))
    stops += ['\\', 'n', '.', ',', '!', '\n']
    beam_scores = torch.zeros((1, num_beams), dtype=torch.float, device='cpu')
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  

    # done sentences
    done = [False for _ in range(1)]

    step = 1
    start_index = len(prompt.split(' '))
    input_ids = [copy.deepcopy(prompt) for _ in range(num_beams)]
    outputs = {}
    _, _, outputs = language_model(prompt, topk=num_beams)
    #prompt = prompt[0]

    print(f"GPT-3 response: {outputs}")
    outputs = outputs[0]
    tokens = list(outputs.keys())
    tokens = [(k, outputs[k]) for k in tokens]

    for i in range(len(tokens)):
        beam_scores[i] = torch.from_numpy(np.array(tokens[i][1]))
        input_ids[i] += tokens[i][0]


    while step < max_length:
        print("STEP:", step)
        full_names, scores, outputs = language_model(input_ids, topk=vocab_size)
        #input_ids = input_ids[0]
        print(f"GPT response: {outputs}")
        scores_ = torch.Tensor([[[omit_(full_names[i][0][j], scores[i][0][j], stops, prompt) for j in range(len(scores[i][0]))]] for i in range(num_beams)])
        scores = scores_.view(num_beams * 1, vocab_size)
        full_names = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(full_names))))
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        next_scores = next_scores.view(1, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        next_tokens_names = [full_names[int(next_tokens[0][i])] for i in range(len(next_tokens[0]))]
        assert next_scores.size()[-1] == len(next_tokens_names) == 2 * num_beams
        classifier_inputs = [classifier.tokenizer.encode(' '.join(input_ids[torch.div(t, vocab_size, rounding_mode="trunc")].split(' ')[start_index:]) + full_names[t]) for t in next_tokens[0]]
        pad_len = max([len(t) for t in classifier_inputs])
        classifier_inputs = torch.LongTensor([b + [0] * (pad_len - len(b)) for b in classifier_inputs])
        logits = torch.nn.functional.log_softmax(classifier(classifier_inputs).logits, 1)[:, (1-mode)].cpu() # Use index 1 if generating benign text
        next_scores = (next_scores * weights[0]) + (logits * weights[1])
        
        next_batch_beam = []
        
        for batch_idx in range(1):
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            next_sent_beam = []

            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):
                beam_id = torch.div(idx, vocab_size, rounding_mode="trunc")
                token_id = full_names[idx]
                effective_beam_id = batch_idx * num_beams + beam_id
                if eos_token_ids is not None and token_id in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id], score.item(),
                    )
                else:
                    next_sent_beam.append((score, token_id, effective_beam_id))

                if len(next_sent_beam) == num_beams:
                    break

            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        assert len(next_batch_beam) == 1 * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = [x[1] for x in next_batch_beam]
        beam_idx = [x[2] for x in next_batch_beam]
        input_ids = [input_ids[i] for i in beam_idx]
        input_ids = [input_ids[i] + beam_tokens[i] for i in range(len(input_ids))]

        if all(done):
            break

        step = step + 1

    for batch_idx in range(1):
        if done[batch_idx]:
            continue

        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    best_all = []
    for i, hypotheses in enumerate(generated_hyps):
        best_all.append(sorted(hypotheses.beams, key=lambda x: x[0],reverse=True))
    return [p[-1] for p in best_all[0]][0]
