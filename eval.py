import sys
import argparse
import torch
import statistics
import numpy as np
import itertools


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from metric.bleus import i_sentence_bleu, i_corpus_bleu
from transformers import AutoTokenizer
from util import build_dd_test_dict_from_csv
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from nltk import word_tokenize


BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]

BLEU_WEIGHTS_SINGLE = [
    [1.0],
    [0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def str_tokenize(sent):
    '''Given a sentence (str), return a list of tokenized characters
    '''
    return word_tokenize(sent)

def generate_fn(model, tokenizer, post):
    '''
    Arguments:
        model (MultiDecoderT5)
        post (str)
    Return:
        tuple (list<str>, list<float>): ([resp1, resp2, ...], [score1, score2, ...])
    '''

    input_ids = tokenizer.encode(post, return_tensors='pt').to(device)

    responses = []
    self_ppls = []

    generated = model.generate(
        input_ids=input_ids,
        # no_repeat_ngram_size=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        repetition_penalty=1.2,  # recommended in https://arxiv.org/pdf/1909.05858.pdf
        output_scores=True,
        return_dict_in_generate=True,
        max_length=20,
    )

    # generated sequence always start with decoder_start_token_id, which we ignore here
    sequence = generated.sequences[0][1:]
    scores = generated.scores
    assert len(sequence) == len(scores)

    log_prob_sum = 0
    for t in range(len(sequence)):
        token_idx = sequence[t]
        log_prob_sum += torch.log_softmax(scores[t][0], dim=0)[token_idx]

    self_ppl = torch.exp(-log_prob_sum / len(sequence)).item()
    response = tokenizer.decode(sequence, skip_special_tokens=True)

    responses.append(response)
    self_ppls.append(self_ppl)

    return responses, self_ppls

def calc_pairwise_bleu(hyps):
    '''Given a list of hypothesis, calculate the pairwise BLEU
    '''
    pairwise_bleu = 0
    perms = list(itertools.permutations(range(len(hyps)), 2))
    for i, j in perms:
        pairwise_bleu += sentence_bleu([hyps[i]], hyps[j])
    return pairwise_bleu / len(perms)

def calculate_ngram_diversity(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity

def eval_model(test_dict, model, tokenizer, generate_func=generate_fn, stream=None):
    '''
    Arguments:
        test_dict (dict): post (str) -> responses (list<str>)
        generate_func (lambda): function that takes a post (str) as input
            and generates a list of responses (list<str>) and their confidences (float)
    Return:
        dict: metric (str) -> value (float)
    '''

    def _log(*args):
        if stream:
            print(*args, file=stream)
        else:
            print(*args)

    chosen_count = np.zeros(1)

    sent_bleu_1s  = []
    sent_bleu_2s  = []
    sent_bleu_3s  = []
    sent_bleu_4s  = []

    sent_ibleu_1s = []
    sent_ibleu_2s = []
    sent_ibleu_3s = []
    sent_ibleu_4s = []

    corp_refs = []  # List[List[List(str)]]
    corp_inps = []  # List[List(str)], list of inputs for iBLEU calcluation

    corp_model_hyps = []  # List[List(str)], list of hypothesis (list of chars)
    corp_best_hyps = []  # List[List(str)], list of hypothesis (list of chars)

    num_posts = len(test_dict)

    for i, (post, reference_responses) in enumerate(test_dict.items()):

        generated_responses, self_ppls = generate_func(model, tokenizer, post)

        inp = str_tokenize(post)
        corp_inps.append(inp)

        ref = list(map(lambda x: str_tokenize(x), reference_responses))
        corp_refs.append(ref)

        # for finding the response that the model is most confident with
        model_response = ''
        lowest_ppl = float('inf')
        chosen_idx = -1

        # for finding the response that works the best
        best_response = ''
        highest_bleu = -1

        _log('{}/{} - Post: {}'.format(i, num_posts - 1, ' '.join(inp)))

        # ----- deal with generated response for each decoder -----
        for j in range(len(generated_responses)):

            generated_response = generated_responses[j]
            self_ppl = self_ppls[j]

            hyp = str_tokenize(generated_response)

            sent_bleu_1 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[0])
            sent_bleu_1s.append(sent_bleu_1)
            sent_bleu_2 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[1])
            sent_bleu_2s.append(sent_bleu_2)
            sent_bleu_3 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[2])
            sent_bleu_3s.append(sent_bleu_3)
            sent_bleu_4 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[3])
            sent_bleu_4s.append(sent_bleu_4)

            sent_ibleu_1 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[0])
            sent_ibleu_1s.append(sent_ibleu_1)
            sent_ibleu_2 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[1])
            sent_ibleu_2s.append(sent_ibleu_2)
            sent_ibleu_3 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[2])
            sent_ibleu_3s.append(sent_ibleu_3)
            sent_ibleu_4 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[3])
            sent_ibleu_4s.append(sent_ibleu_4)

            bleu = sent_bleu_4

            if bleu > highest_bleu:
                highest_bleu = bleu
                best_response = generated_response
            if self_ppl < lowest_ppl:
                lowest_ppl = self_ppl
                model_response = generated_response
                chosen_idx = j

            _log('Decoder {}, bleu={:.5f}, self_ppl={:9.2f}: {}'.format(j, bleu, self_ppl, generated_response))
            _log('Ref Response: {}'.format(reference_responses[0]))

        chosen_count[chosen_idx] += 1

        _log()

        corp_model_hyps.append(str_tokenize(model_response))
        corp_best_hyps.append(str_tokenize(best_response))

    _log('---------- Results ----------')

    # print(i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps))
    _log('sent_bleus (1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        statistics.mean(sent_bleu_1s),
        statistics.mean(sent_bleu_2s),
        statistics.mean(sent_bleu_3s),
        statistics.mean(sent_bleu_4s),
    ))
    _log('sent_ibleus (1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        statistics.mean(sent_ibleu_1s),
        statistics.mean(sent_ibleu_2s),
        statistics.mean(sent_ibleu_3s),
        statistics.mean(sent_ibleu_4s),
    ))
    _log()

    corp_model_bleu1 = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[0])
    corp_model_bleu = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[3])
    _log('corp_model_bleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        corp_model_bleu1,
        corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[1]),
        corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[2]),
        corp_model_bleu,
    ))

    corp_model_ibleu1 = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[0])
    corp_model_ibleu = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[3])
    _log('corp_model_ibleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        corp_model_ibleu1,
        i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[1]),
        i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[2]),
        corp_model_ibleu,
    ))
    _log()

    corp_best_bleu = corpus_bleu(corp_refs, corp_best_hyps, weights=BLEU_WEIGHTS_MEAN[3])
    _log('corp_best_bleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        corpus_bleu(corp_refs, corp_best_hyps, weights=BLEU_WEIGHTS_MEAN[0]),
        corpus_bleu(corp_refs, corp_best_hyps, weights=BLEU_WEIGHTS_MEAN[1]),
        corpus_bleu(corp_refs, corp_best_hyps, weights=BLEU_WEIGHTS_MEAN[2]),
        corp_best_bleu,
    ))
    corp_best_ibleu = i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[3])
    _log('corp_best_ibleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[0]),
        i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[1]),
        i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[2]),
        corp_best_ibleu,
    ))

    tokens = [token for sentence in corp_model_hyps for token in sentence]
    dist_1, dist_2 = calculate_ngram_diversity(tokens)
    _log('dist_1: {:.5f}, dist_2: {:.5f}'.format(dist_1, dist_2))

    _log()

    # eval_ as prefix for huggingface logger to understand that this is eval...
    return {
        'corp_model_bleu1': corp_model_bleu1,
        'corp_model_bleu': corp_model_bleu,
        'corp_model_ibleu1': corp_model_ibleu1,
        'corp_model_ibleu': corp_model_ibleu,
        'corp_best_bleu': corp_best_bleu,
        'corp_best_ibleu': corp_best_ibleu,
        'dist_1': dist_1,
        'dist_2': dist_2,
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Script for evaluating models')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--test-dict-path', type=str, default='data/hareesh/df_daily_test_without_duplicates.csv')
    parser.add_argument('--max-num-dialogues', type=int, default=sys.maxsize)

    args = parser.parse_args()

    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    test_dict = build_dd_test_dict_from_csv(
        path=args.test_dict_path,
        max_num_dialogues=args.max_num_dialogues,
    )

    if args.output_file:
        stream = open(args.output_file, mode='w')
    else:
        stream = open(args.ckpt + '.test', mode='w')

    eval_model(test_dict, model, tokenizer, stream=stream)
