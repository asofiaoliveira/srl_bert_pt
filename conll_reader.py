from typing import DefaultDict, List, Optional, Iterator, Set, Tuple, Dict, Any, Iterable
from collections import defaultdict
import codecs
import os
import logging
from nltk import Tree

from overrides import overrides
from transformers import AutoTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedSpan
from allennlp_models.common.ontonotes import OntonotesSentence
from allennlp_models.syntax.srl.srl_reader import _convert_verb_indices_to_wordpiece_indices, _convert_tags_to_wordpiece_tags

# This file is very similar to allennlp_models/syntax/srl/srl_reader.py and allennlp_models/common/ontonotes.py
# The changes are due to the needed preprocessing for the English CoNLL-2012 data
# to match the Portuguese data

class Ontonotes_mine:
    def __init__(self):
        self.flag = []

    def dataset_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntonotesSentence]]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with codecs.open(file_path, "r", encoding="utf8") as open_file:
            conll_rows = []
            document: List[OntonotesSentence] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        sentence = self._conll_rows_to_sentence(conll_rows)
                        if len(self.flag) != 0: 
                            # Remove instances that have tags we don't care about
                            sentence.srl_frames = [srl_frame for ind, srl_frame in enumerate(sentence.srl_frames) if ind not in self.flag]
                            self.flag = []
                            if len(sentence.srl_frames) == 0:
                                # If there are no annotated verbs left
                                conll_rows = []
                                continue
                        document.append(sentence)
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        document_id: str = None
        sentence_id: int = None
        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        parse_pieces: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: List[str] = []
        # The sense of the word, if available.
        word_senses: List[float] = []
        # The current speaker, if available.
        speakers: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

        for index, row in enumerate(conll_rows):
            conll_components = row.split()

            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]

            # Replace brackets in text and pos tags
            # with a different token for parse trees.
            if pos_tag != "XX" and word != "XX":
                if word == "(":
                    parse_word = "-LRB-"
                elif word == ")":
                    parse_word = "-RRB-"
                else:
                    parse_word = word
                if pos_tag == "(":
                    pos_tag = "-LRB-"
                if pos_tag == ")":
                    pos_tag = "-RRB-"
                (left_brackets, right_hand_side) = parse_piece.split("*")
                # only keep ')' if there are nested brackets with nothing in them.
                right_brackets = right_hand_side.count(")") * ")"
                parse_piece = f"{left_brackets} ({pos_tag} {parse_word}) {right_brackets}"
            else:
                # There are some bad annotations in the CONLL data.
                # They contain no information, so to make this explicit,
                # we just set the parse piece to be None which will result
                # in the overall parse tree being None.
                parse_piece = None

            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]

            self._process_span_annotations_for_word(
                conll_components[10:-1], span_labels, current_span_labels
            )

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any("(V" in x for x in conll_components[11:-1])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            self._process_coref_span_annotations_for_word(
                conll_components[-1], index, clusters, coref_stacks
            )

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != "-" else None)
            word_senses.append(float(word_sense) if word_sense != "-" else None)
            speakers.append(speaker if speaker != "-" else None)

        named_entities = span_labels[0]

        for j in range(1,len(span_labels)):
            labs=["A0","A1","A2","A3","A4","A5"]
            if any(x in span_labels[j] for x in ["B-R-A0","B-R-A1","B-R-A2","B-R-A3","B-R-A4","B-R-A5"]):
                for lab in labs:
                    if "B-R-"+lab in span_labels[j]:
                        for i in range(len(span_labels[j])):
                            if lab in span_labels[j][i] and "R" not in span_labels[j][i]:
                                span_labels[j][i] = "O"
                            elif "R-"+lab in span_labels[j][i]:
                                span_labels[j][i] = span_labels[j][i][0]+span_labels[j][i][3:]
                    

        srl_frames = [
            (predicate, labels) for predicate, labels in zip(verbal_predicates, span_labels[1:])
        ]
        

        if all(parse_pieces):
            parse_tree = Tree.fromstring("".join(parse_pieces))
        else:
            parse_tree = None
        coref_span_tuples: Set[TypedSpan] = {
            (cluster_id, span) for cluster_id, span_list in clusters.items() for span in span_list
        }
        return OntonotesSentence(
            document_id,
            sentence_id,
            sentence,
            pos_tags,
            parse_tree,
            predicate_lemmas,
            predicate_framenet_ids,
            word_senses,
            speakers,
            named_entities,
            srl_frames,
            coref_span_tuples,
        )

    @staticmethod
    def _process_coref_span_annotations_for_word(
        label: str,
        word_index: int,
        clusters: DefaultDict[int, List[Tuple[int, int]]],
        coref_stacks: DefaultDict[int, List[int]],
    ) -> None:
        """
        For a given coref label, add it to a currently open span(s), complete a span(s) or
        ignore it, if it is outside of all spans. This method mutates the clusters and coref_stacks
        dictionaries.

        # Parameters

        label : `str`
            The coref label for this word.
        word_index : `int`
            The word index into the sentence.
        clusters : `DefaultDict[int, List[Tuple[int, int]]]`
            A dictionary mapping cluster ids to lists of inclusive spans into the
            sentence.
        coref_stacks : `DefaultDict[int, List[int]]`
            Stacks for each cluster id to hold the start indices of active spans (spans
            which we are inside of when processing a given word). Spans with the same id
            can be nested, which is why we collect these opening spans on a stack, e.g:

            [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1
        """
        if label != "-":
            for segment in label.split("|"):
                # The conll representation of coref spans allows spans to
                # overlap. If spans end or begin at the same word, they are
                # separated by a "|".
                if segment[0] == "(":
                    # The span begins at this word.
                    if segment[-1] == ")":
                        # The span begins and ends at this word (single word span).
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        # The span is starting, so we record the index of the word.
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    # The span for this id is ending, but didn't start at this word.
                    # Retrieve the start index from the document state and
                    # add the span to the clusters for this id.
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))

   
    def _process_span_annotations_for_word(
        self,
        annotations: List[str],
        span_labels: List[List[str]],
        current_span_labels: List[Optional[str]],
    ) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        # Parameters

        annotations : `List[str]`
            A list of labels to compute BIO tags for.
        span_labels : `List[List[str]]`
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : `List[Optional[str]]`
            The currently open span per annotation type, or `None` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")
            
            if annotation_index != 0:
                if "C-" in label or "R-" in label:
                    label = label[:3]+label[5:]
                elif "A" in label:
                    label = label[0] + label[3:]
            if label in ["AM-ADJ","R-AM-LOC","R-AM-TMP","AM-LVB","R-AM-MNR","AM-DSP","AA","R-AM-CAU","R-AM-ADV","R-AM-DIR","R-AM-PRP","R-AM-EXT","AM-PRR","AM-PRX","R-AM-GOL","R-AM-PNC","R-AM-COM","R-AM-PRD","R-AM-MOD"]:
                span_labels[annotation_index].append("O")
                continue
            
            if label == "AM-PNC":
                label = "AM-PRP"
            if label == "C-AM-PNC":
                label = "C-AM-PRP"
            
            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label                    
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None




@DatasetReader.register("srl_mine")
class SrlMineReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : `TextField`
        The tokens in the sentence.
    verb_indicator : `SequenceLabelField`
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : `SequenceLabelField`
        A sequence of Propbank tags for the given verb in a BIO format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    domain_identifier : `str`, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : `Optional[str]`, (default = None)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.

    # Returns

    A `Dataset` of `Instances` for Semantic Role Labelling.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self.xlm = "xlm" in bert_model_name
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        if self.xlm:
            self.vocab = {self.bert_tokenizer.convert_ids_to_tokens(i): i for i in range(250001)}
            self.vocab.update(self.bert_tokenizer.added_tokens_encoder)

        self.lowercase_input = False

    def _wordpiece_tokenize_input(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
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

        # Returns

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
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes_mine()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(
                "Filtering to only include file paths containing the %s domain",
                self._domain_identifier,
            )

        for sentence in self._ontonotes_subset(
            ontonotes_reader, file_path, self._domain_identifier
        ):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                s =  self.text_to_instance(tokens, verb_label, tags)
                if s is None:
                    continue
                yield s

            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    s = self.text_to_instance(tokens, verb_indicator, tags)
                    if s is None:
                        continue
                    yield s

    @staticmethod
    def _ontonotes_subset(
        ontonotes_reader: Ontonotes_mine, file_path: str, domain_identifier: str
    ) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

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
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator

        if all(x == 0 for x in verb_label):
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
                fields["tags"] = SequenceLabelField(new_tags, text_field)
            else:
                fields["tags"] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        
        if len(wordpieces)>190:
            logger.info("%s %i" % (wordpieces, len(wordpieces)))
            return None
        return Instance(fields)

