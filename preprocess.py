from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
import logging


class Preprocess():
    """ 
    Preprocess the input data according to the steps described in Daniel Falci's thesis
    Inputs:
    Parameters
    ----------
    tokens : ``List[str]``
        List of instance tokens.
    tags: ``List[str]``, (default = None)
        List of the associated tags for this instance. 
        When predicting, this field isn't used.
    separate : ``bool``, (default = True)
        Whether to separate the words joined by "_".   
    contract : ``bool``, (default = True)
        Whether to perform word contractions (e.g.: "de"+"o" -> "do")

    """
    def __init__(self,
                 tokens: List[str],
                 tags: List[str] = None,
                 separate: bool = True, 
                 contract: bool = True
    ) -> None:
        self.tokens = tokens
        self.tags = tags
        self.has_tags = tags!=None
        self.art_def = ['a', 'as', 'o', 'os']
        self.art_def_masc = ['o', 'os']
        self.adverb = ['aí', 'aqui', 'ali'] 
        self.pron = ['ele', 'eles', 'ela', 'elas', 'esse', 'esses', 'essa', 'essas', 'isso', 'este', 'estes', 'esta', 'estas', 'isto'] 
        self.pron2 = ['aquele', 'aqueles', 'aquela', 'aquelas', 'aquilo'] 
        self.ref = ['o', 'os', 'a', 'as', 'me', 'se', 'te', 'vos', 'lhe', 'lho', 'lhas', 'lhos', 'lha', 'lo', 'la', 'los', 'las', 'lhes', 'no', 'na', 'nos']
        self.change = self.art_def + self.adverb + self.pron + self.pron2 + self.ref
        self.prepositions = ['de', 'a', 'em', 'por', '-']
        self._separate = separate
        self._contract = contract

    def preprocess(self):
        if self._separate:
            self.separate()
        if self._contract:
            self.contractions()
        #Check that tokens and tags list has same size
        if self.has_tags and len(self.tokens) != len(self.tags):
            raise Exception("Lengths of token list %i and tag list %i don't match." % (len(self.tokens), len(self.tags)))
        return self.tokens, self.tags

    def separate(self):
        """
        We want to separate multiword nouns and expressions such as 
        "Campeonato_Brasileiro" and "em_termos_de" into several tokens
        (removing the "_")
        """
        ind = 0
        while ind < len(self.tokens):
            if '_' in self.tokens[ind] and '_' != self.tokens[ind]: #sometimes '_' represents '--'
                new_tokens = self.tokens[ind].split('_')
                self.tokens = self.tokens[:ind] + new_tokens + self.tokens[ind+1:]
                if self.has_tags:
                    if self.tags[ind][:2] == 'B-':
                        self.tags = self.tags[:ind+1] + ['I-' + self.tags[ind][2:]]*(len(new_tokens)-1) + self.tags[ind+1:]
                    else:
                        self.tags = self.tags[:ind] + [self.tags[ind]]*len(new_tokens) + self.tags[ind+1:] 
            ind += 1

    def determine_tags(self, ind, new_token):
        """
        Given a new, contracted token at position ind, 
        change tags list to reflect the new tokens list.
        """
        if self.tags[ind] == self.tags[ind+1]:
            del self.tags[ind]
        elif self.tags[ind] == 'O' and self.tags[ind+1] != 'O':
            del self.tags[ind]
        elif self.tags[ind] != 'O' and self.tags[ind+1] == 'O':
            del self.tags[ind+1]
            if "C-" in self.tags[ind+1] and self.tags[ind+1][4:] == self.tags[ind][2:]:
                for i, tag in enumerate(self.tags[ind+1:],ind+1):
                    if "C-" + self.tags[ind][2:] in tag:
                        self.tags[i] = "I-" + tag[4:]
        elif self.tags[ind][:2] == "B-" and self.tags[ind+1][:2] == "I-" and self.tags[ind][2:] == self.tags[ind+1][2:]:
            del self.tags[ind+1]
        else:
            raise Exception()

    def contractions(self):
        """
        We want to contract prepositions and articles, pronouns and adverbs
        These contractions are widely used in the portuguese language and a lot of times
        are used instead of the separated form even in formal settings
        """
        if any(token in self.change for token in self.tokens) and any(token.lower() in self.prepositions for token in self.tokens):
            ind = 0
            while ind < len(self.tokens)-1:
                # The only captalization that matters is in the first word of the contraction
                is_cap = self.tokens[ind][0].isupper()

                prev_token = self.tokens[ind].lower()
                cur_token = self.tokens[ind+1].lower()

                if prev_token in self.prepositions and cur_token in self.change:
                    if prev_token == 'de' and cur_token in self.art_def + self.adverb + self.pron + self.pron2: 
                        new_token = 'd' + cur_token
                    elif prev_token == 'em' and cur_token in self.art_def + self.pron + self.pron2:
                        new_token = 'n' + cur_token
                    elif prev_token == 'a' and cur_token in self.art_def + self.pron2 and cur_token not in self.art_def_masc:
                        new_token = 'à' + cur_token[1:]
                    elif prev_token == 'a' and cur_token in self.art_def_masc:
                        new_token = 'a' + cur_token
                    elif prev_token == 'por' and cur_token in self.art_def:
                        new_token = 'pel' + cur_token
                    elif prev_token == '-' and cur_token in self.ref:
                        new_token = '-' + cur_token
                    else:
                        ind += 1
                        continue
                    
                    if is_cap:
                        new_token = new_token.capitalize()
                    self.tokens = self.tokens[:ind] + [new_token] + self.tokens[ind+2:]
        
                    if self.has_tags:
                        self.determine_tags(ind, new_token)
                ind += 1
                
