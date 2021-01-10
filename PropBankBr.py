from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
import logging
import random

from nltk import Tree

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# This file is similar to ontonotes.py
# in allennlp_models/common, but reads the portuguese
# CoNLL files (some columns are different from ontonotes)

class PropBankBrSentence:
    """
    A class representing the annotations available for a single PropBank.Br formatted sentence.

    Parameters
    ----------
    words : ``List[str]``
        This is the tokens as segmented/tokenized in the Treebank.
    predicate_lemmas : ``List[Optional[str]]``
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are ``None``.
    predicate_senses : ``List[Optional[int]]``
        The VerbNet PT frameset ID of the lemmas in ``predicate_lemmas``, or ``None``.
    srl_frames : ``List[Tuple[str, List[str]]]``
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    """
    def __init__(self,
                 words: List[str],
                 predicate_senses: List[Optional[str]],
                 predicate_lemmas: List[Optional[str]],
                 srl_frames: List[Tuple[str, List[str]]]) -> None:

        self.words = words
        self.predicate_senses = predicate_senses
        self.predicate_lemmas = predicate_lemmas
        self.srl_frames = srl_frames


class PropBankBr:
    """
    This DatasetReader is designed to read in the PropBank-Br v1.1 data. 

    The file path provided to this class can then be any of the train, test or development
    directories.

    The important columns are:
    1 Word : ``str``
        This is the token as segmented/tokenized in the Treebank. 
    8 Predicate Frameset ID: ``int``
        The PropBank frameset ID of the predicate in Column 7.
    9 Predicate lemma: ``str``
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    10+ Predicate Arguments: ``str``
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    """
    def __init__(self, remove_c):
        self.flag = []
        # Flag to remove the continuation arguments (to compare with Falci et al.):
        self.remove_c = remove_c

    def dataset_iterator(self, file_path: str) -> Iterator[PropBankBrSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
                
        for sentence in self.dataset_document_iterator(file_path):
            yield sentence

    def dataset_document_iterator(self, file_path: str) -> List[PropBankBrSentence]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with codecs.open(file_path, 'r', encoding='UTF-8') as open_file:
            conll_rows = []
            document: List[PropBankBrSentence] = []
            for line in open_file:
                line = line.strip()
                if line != '' and not line.startswith('#'):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        sentence = self._conll_rows_to_sentence(conll_rows)
                        if len(self.flag) != 0: 
                            # Remove instances that have either C- tags 
                            # (if we want to remove them) or double labels
                            sentence.srl_frames = [srl_frame for ind, srl_frame in enumerate(sentence.srl_frames) if ind not in self.flag]
                            self.flag = []
                            if len(sentence.srl_frames) == 0:
                                # If there are no annotated verbs left
                                conll_rows = []
                                continue
                        document.append(sentence)
                        conll_rows = []
                if line.startswith("#end document"):
                    return document
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                return document


    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> PropBankBrSentence:

        # The words in the sentence.
        sentence: List[str] = []
        # The VerboBrasil ID of the predicate.
        predicate_senses: List[str] = []
        # The lemmatised form of the predicates in the sentence 
        predicate_lemmas: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        for _, row in enumerate(conll_rows):
            conll_components = row.split()

            word = conll_components[1]
            lemmatised_predicate = conll_components[9]
            sense = conll_components[8]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:]]

            self._process_span_annotations_for_word(self,
                                                    conll_components[10:],
                                                    span_labels,
                                                    current_span_labels)

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            # !! perhaps change this to only adding if it has a predicate sense
            word_is_verbal_predicate = any(["(V" in x for x in conll_components[10:]])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            sentence.append(word)
            predicate_lemmas.append(lemmatised_predicate if lemmatised_predicate != "-" else None)
            predicate_senses.append(sense if sense != "-" else None)

        srl_frames = [(predicate, labels) for predicate, labels
                      in zip(verbal_predicates, span_labels)]
        return PropBankBrSentence(sentence,
                                 predicate_senses,
                                 predicate_lemmas,
                                 srl_frames)

    @staticmethod
    def _process_span_annotations_for_word(self,
                                           annotations: List[str],
                                           span_labels: List[List[str]],
                                           current_span_labels: List[Optional[str]]) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        Parameters
        ----------
        annotations: ``List[str]``
            A list of labels to compute BIO tags for.
        span_labels : ``List[List[str]]``
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : ``List[Optional[str]]``
            The currently open span per annotation type, or ``None`` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")
            if self.remove_c and  "C-" in label:
                self.flag.append(annotation_index)
            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                if current_span_labels[annotation_index] is not None:
                    # If there is already a current label but the label has a "(", 
                    # this token has 2 annotations and it is to be considered an annotation error.
                    # Do not append to the document.
                    self.flag.append(annotation_index)
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
