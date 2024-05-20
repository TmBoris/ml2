from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import xml.etree.ElementTree as ET



@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    # это чзх вообще, как я догадаться должен был. Спасибо человеку в беседе.
    with open(filename, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('&', '&amp;')

    with open(filename, 'w') as file:
        file.write(filedata)

    # норм код
    sentence_pairs = []
    alignments = []
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for s_element in root:
        english_sentence = s_element.find('english').text.split()
        czech_sentence = s_element.find('czech').text.split()
        sentence_pair = SentencePair(english_sentence, czech_sentence)
        sentence_pairs.append(sentence_pair)
        
        sure_alignments = []
        possible_alignments = []
        sure_element = s_element.find('sure').text
        possible_element = s_element.find('possible').text
        if sure_element is not None:
            sure_alignments = [tuple(map(int, pair.split('-'))) for pair in sure_element.split()]
        if possible_element is not None:
            possible_alignments = [tuple(map(int, pair.split('-'))) for pair in possible_element.split()]

        labeled_alignment = LabeledAlignment(sure_alignments, possible_alignments)
        alignments.append(labeled_alignment)
    
    return sentence_pairs, alignments



def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    en_counter = Counter()
    cz_counter = Counter()

    for sent_pair in sentence_pairs:
        for word in sent_pair.source:
            en_counter[word] += 1
        for word in sent_pair.target:
            cz_counter[word] += 1

    if freq_cutoff is not None:
        en_vocab = {word[0]: ind for ind, word in enumerate(en_counter)}
        cz_vocab = {word[0]: ind for ind, word in enumerate(cz_counter)}
    else:
        en_vocab = {word[0]: ind for ind, word in enumerate(en_counter.most_common(freq_cutoff))}
        cz_vocab = {word[0]: ind for ind, word in enumerate(cz_counter.most_common(freq_cutoff))}

    return en_vocab, cz_vocab


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    ans = []

    for sent_pair in sentence_pairs:
        new_src = np.array([source_dict[word] for word in sent_pair.source if word in source_dict])
        new_trg = np.array([target_dict[word] for word in sent_pair.target if word in target_dict])
        if new_src.size > 0 and new_trg.size > 0:
            ans.append(TokenizedSentencePair(new_src, new_trg))

    return ans

