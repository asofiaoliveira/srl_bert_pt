import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from transformers import AutoTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from PropBankBr import PropBankBr, PropBankBrSentence
from allennlp_models.structured_prediction.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from conllu import parse_incr

import preprocess
import codecs

# This file is heavily based on the dataset readers from allennlp_models
# But the original file couldn't be used with xlmr models (at the time)


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]):
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]



@DatasetReader.register("srl-pt")
class SrlReader(DatasetReader):
    """
    This dataset reader is almost identical to allennlp_models/syntax/srl/srl_reader.py
    But the original file couldn't be used with xlmr models (at the time)
    And the _read was changed to read folds data.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : ``Optional[str]``, (default = None)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self.xlm = "xlm" in bert_model_name
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # the model class is different in xlmr models 
        # the vocab is not in the same place as in bert
        # so we create the vocab in the xlmr
        if self.xlm:
            self.vocab = {self.bert_tokenizer.convert_ids_to_tokens(i): i for i in range(250001)}
            self.vocab.update(self.bert_tokenizer.added_tokens_encoder)

        self.lowercase_input = False
        
    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.xlm:
                word_pieces = self.bert_tokenizer._tokenize(token)
            else:
                word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        return word_piece_tokens, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # This reads the folds data
        
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading SRL instances from dataset files at: %s", file_path)

        with codecs.open(file_path, 'r', encoding='utf8') as open_file:
            for line in open_file:
                words, tags = line.split('\t')
                words, tags = words.split(' '), tags.strip().split(' ')
                verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                tokens = [Token(t) for t in words]
                yield self.text_to_instance(tokens, verb_indicator, tags)
                

            

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
            
            if self.xlm:
                wordpieces = ["<s>"] + wordpieces + ["</s>"]
            else:
                wordpieces = ["[CLS]"] + wordpieces + ["[SEP]"]
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            if self.xlm: 
                text_field = TextField([Token(t, text_id=self.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            else:
                text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        fields['verb_indicator'] = verb_indicator

        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags
        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)


@DatasetReader.register("simple")
class SimpleReader(DatasetReader):
    """
    This data set reader is meant to read conll data, 
    preprocess it and yield it to create the folds.
    The data output is meant to be written in folds files,
    which will the be read by the SrlReader above.
    """
    def __init__(self,
                 lazy: bool = False, 
                 remove_c: bool = False) -> None:
        super().__init__(lazy)
        self.remove_c = remove_c

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        PBBr_reader = PropBankBr(remove_c = self.remove_c)
        logger.info("Reading SRL instances from dataset files at: %s", file_path)

        for sentence in PBBr_reader.dataset_iterator(file_path):
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in sentence.words]
                words, tags = preprocess.Preprocess(sentence.words, tags).preprocess()
                yield [words, tags]
            else:
                for (_, tags) in sentence.srl_frames:
                    words, tags = preprocess.Preprocess(sentence.words, tags).preprocess()
                    yield [words, tags]



@DatasetReader.register("ud_transformer", exist_ok=True)
class UDTransformerDatasetReader(UniversalDependenciesDatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : `bool`, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        tokenizer: Tokenizer = None,
        bert_model_name:str = None,
        **kwargs,
    ) -> None:
        super().__init__(token_indexers, use_language_specific_pos, tokenizer, **kwargs)
        self.xlm = "xlm" in bert_model_name
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        if self.xlm:
            self.vocab = {self.bert_tokenizer.convert_ids_to_tokens(i): i for i in range(250001)}
            self.vocab.update(self.bert_tokenizer.added_tokens_encoder)

        self.lowercase_input = False

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r", encoding="utf-8") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = [x["head"] for x in annotation]
                tags = ["B-"+x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = ["B-"+x["upostag"] for x in annotation]
                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)))

        
    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.xlm:
                word_pieces = self.bert_tokenizer._tokenize(token)
            else:
                word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative+1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        return word_piece_tokens, end_offsets, start_offsets


    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
    ) -> Instance:

        """
        # Parameters

        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t for t in words])
            if self.xlm:
                wordpieces = ["<s>"] + wordpieces + ["</s>"]
            else:
                wordpieces = ["[CLS]"] + wordpieces + ["[SEP]"]
            if self.xlm: 
                text_field = TextField([Token(t, text_id=self.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            else:
                text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
        else:
            tokens = [Token(t) for t in words]
            text_field = TextField(tokens, token_indexers=self._token_indexers)
        
        fields["words"] = text_field
        fields["pos_tags"] = SequenceLabelField(_convert_tags_to_wordpiece_tags(upos_tags, offsets), text_field, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                _convert_tags_to_wordpiece_tags([x[0] for x in dependencies],offsets), text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                self._util([x[1] for x in dependencies],offsets), text_field, label_namespace="head_index_tags"
            )

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags, "offsets": start_offsets})
        return Instance(fields)

    def _util(self, tags, offsets):
        new_tags = []
        j = 0
        mapping = {}
        for i, offset in enumerate(offsets):
            tag = tags[i]
            is_start = True
            while j < offset:
                if is_start:
                    mapping[i+1] = len(new_tags)+1
                    new_tags.append(tag)
                    is_start = False
                else:
                    new_tags.append(i+1)
                j += 1
        mapping[0] = 0
        n = []
        for tag in new_tags:
            n.append(mapping[tag] if tag in mapping.keys() else tag)
        # Add O tags for cls and sep tokens.)
        return [0] + n + [0]
