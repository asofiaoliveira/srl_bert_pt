import logging
import sys
import os.path
import json
import re
import torch
import torch.nn.init
from typing import Any, Dict
from torch.nn import Embedding

from overrides import overrides

from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate

from allennlp.common import Params, FromParams, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Initializer
from allennlp.commands.train import train_model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder


import my_reader
import PropBankBr
import my_model
import conll_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)  


@TokenEmbedder.register("pretrained_transformer_mine")
class PretrainedTransformerEmbedderMine(PretrainedTransformerEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
    ) -> None:
        super().__init__(model_name,
                        max_length,
                        sub_module,
                        train_parameters)
        self.transformer_model.config.type_vocab_size = 2 
            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
        self.transformer_model.embeddings.token_type_embeddings = Embedding(2, self.transformer_model.config.hidden_size)
            # Initialize it
        self.transformer_model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.transformer_model.config.initializer_range)
        self.config = self.transformer_model.config


@Initializer.register("my_initializer")
class MyInitializer(Initializer):
    """
    Very similar to PretrainedModelInitializer by allennlp.
    Difference is that it matches the parameter names to override using regex.
    """
 
    def __init__(
        self, weights_file_path: str, parameter_name_overrides: Dict[str, str] = None
    ) -> None:
        self.weights: Dict[str, torch.Tensor] = torch.load(weights_file_path)
        self.parameter_name_overrides = parameter_name_overrides or {}

    @overrides
    def __call__(self, tensor: torch.Tensor, parameter_name: str, **kwargs) -> None:  # type: ignore
        # Select the new parameter name if it's being overridden
        k = list(self.parameter_name_overrides.keys())[0]
        #I needed regex here, so as not to have to write every single parameter
        if re.match(k,parameter_name):
            parameter_name = re.sub(k, self.parameter_name_overrides[k], parameter_name)

        # If the size of the source and destination tensors are not the
        # same, then we need to raise an error
        source_weights = self.weights[parameter_name]
        if tensor.data.size() != source_weights.size():
            raise ConfigurationError(
                "Incompatible sizes found for parameter %s. "
                "Found %s and %s" % (parameter_name, tensor.data.size(), source_weights.size())
            )
        # Copy the parameters from the source to the destination
        tensor.data[:] = source_weights[:]


class Train():
    """
    This is the class that will train the models. It uses the data created by create_folds.py
    It also evaluates the obtained models in the appropriate test set, and optionally
    on the Buscapé set, using the Evaluate class.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    buscape: ``bool``, (default = True)
        Whether to test the models in the Buscapé data.
    cont : ``bool``, (default = False)
        Parameter to pass to the evaluation function.
        Signals whether the model was previously trained in English SRL.
        This is needed to pass the overrides parameter when evaluating.
        
    """
    def __init__(self, params: Params, buscape: bool = True, cont: bool = False):
        self.params = params
        self.buscape = buscape
        self.cont = cont

    def train_fold(self,
                   serialization_dir: str,
                   train_data_path: str, 
                   validation_data_path: str,
                   test_data_path: str,
                   fold_ind: int = 0):

        """
        This is the training function for one fold. 
        It trains the model (defined in params) and stores the results in 
        serialization_dir.
        If there is a test file for this fold, it evaluates the trained model in it.
        If self.buscape is True, it evaluates the trained model in the Buscapé data.

        Parameters
        ----------
        serialization_dir : ``str``
            The folder where the results of the training and testing will be stored.
        train_data_path : ``str``
            Path to training data.
        validation_data_path : ``str``
            Path to validation data.
        test_data_path : ``str``
            Path to test data.
        fold_ind: ``int``, (default = 0)
            The number of the current fold.
        """
        # Set the data paths in the Params object

        params_to_pass = params.duplicate()
        params_to_pass.__setitem__("train_data_path", train_data_path)
        params_to_pass.__setitem__("validation_data_path", validation_data_path)
        
        if os.path.isfile(test_data_path):
            params_to_pass.__setitem__("test_data_path", test_data_path)

        train_model(params = params_to_pass,
                    serialization_dir = serialization_dir)  

        #if os.path.isfile(test_data_path):
        #    # in the first experiment, there may not be a test file
        #    output_file = "test" + str(fold_ind) + ".txt"

            # this file_path is then imported by my_model during testing 
            # to save the list of tags associated with the predictions
        #    file_path = os.path.dirname(serialization_dir) + "/tags_" + output_file
        #    e = Evaluate(serialization_dir, self.cont)      
        #    e.evaluate(test_data_path, os.path.dirname(serialization_dir) + "/metrics_" + output_file)

        #if self.buscape:
        #    output_file = "buscape" + str(fold_ind) + ".txt"
        #    file_path = os.path.dirname(serialization_dir) + "/tags_" + output_file
        #    e.evaluate("buscape/test0.txt", os.path.dirname(serialization_dir) + "/metrics_" + output_file)

    def iterate_folds(self):
        """
        Training function for a model. Iterates through the folds existing in folds_dir.
        """
        n_folds = ["train" in i for i in os.listdir(folds_dir)].count(True)
        for fold_ind in range(n_folds):
            logger.info("Training fold number %s", fold_ind+1)
            serialization_dir = ser_dir + "/results_fold" + str(fold_ind)
            self.train_fold(serialization_dir = serialization_dir,
                            train_data_path = folds_dir + "/train" + str(fold_ind) + ".txt",
                            validation_data_path = folds_dir + "/val" + str(fold_ind) + ".txt",
                            test_data_path = folds_dir + "/test" + str(fold_ind) + ".txt",
                            fold_ind = fold_ind)
            


class Evaluate():
    """
    This class evaluates the models and stores the results in the training folder.
    Heavily based on allennlp's evaluation function
    Parameters
        ----------
        model_path : ``str``
            Path to trained model.
        cont : ``bool``
            Signals whether the model was previously trained in English SRL.
            This is needed to pass the overrides parameter when evaluating.
        
    """
    def __init__(self, model_path: str, cont: bool = False):
        self.model_path = model_path
        self.cont = cont
        logging.getLogger("allennlp.common.params").disabled = True
        logging.getLogger("allennlp.nn.initializers").disabled = True
        logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    def evaluate(self, evaluation_data_path: str, output_file: str, validation_dataset_reader_params: str = None):
        # Load from archive
        if self.cont:
            over = "" 
        else:
            over = "{\"model.initializer\":{\"regexes\":[]}}" 
        archive = load_archive(self.model_path, overrides = over)
        config = archive.config
        
        if validation_dataset_reader_params is not None:
            config.__setitem__("validation_dataset_reader", validation_dataset_reader_params)
        prepare_environment(config)
        model = archive.model
        model.eval()

        # Load the evaluation data

        # Try to use the validation dataset reader if there is one - otherwise fall back
        # to the default dataset_reader used for both training and validation.
        validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
        if validation_dataset_reader_params is not None:
            dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
        else:
            dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
        
        logger.info("Reading evaluation data from %s", evaluation_data_path)
        instances = dataset_reader.read(evaluation_data_path)

        instances.index_with(model.vocab)
        data_loader_params = config.pop("validation_data_loader", None)
        if data_loader_params is None:
            data_loader_params = config.pop("data_loader")
        data_loader = DataLoader.from_params(dataset=instances, params=data_loader_params)

        metrics = evaluate(model, data_loader, -1, "")

        logger.info("Finished evaluating.")

        dump_metrics(output_file, metrics, log=True)



if __name__ == "__main__":
    folds_dir = "data/folds_dfalci_20"
    for model_type in ["brbert-large","brbert-base"]:
        ser_dir = "results_" + model_type + "_dfalci"
        PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
        params = Params.from_file(PARAMS_FILE)
        Train(params, False).iterate_folds()

    for model_type in ["xlmr-base_conll", "xlmr-large_conll", "mbert_conll"]:
        ser_dir = "results_" + model_type 
        PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
        params = Params.from_file(PARAMS_FILE)
        Train(params, False).train_fold(ser_dir, 
                                "data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train",
                                "data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development",
                                "")

    #Train ud models
    for model_type in ["brbert-large_ud_pre", "xlmr-large_ud_pre"]:
        ser_dir = "results_" + model_type 
        PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
        params = Params.from_file(PARAMS_FILE)
        Train(params, False).train_fold(ser_dir, 
                                "data/ud/pt_bosque-ud-train.conllu.txt",
                                "data/ud/pt_bosque-ud-dev.conllu.txt",
                                "")

    #Train ud + conll (english srl only)
    model_type = "xlmr-large_ud_conll"
    ser_dir = "results_" + model_type 
    PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
    params = Params.from_file(PARAMS_FILE)
    Train(params, False).train_fold(ser_dir, 
                                "data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train",
                                "data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development",
                                "")

    folds_dir = "data/folds_10"
    for model_type in ["xlmr-base_en", "xlmr-large_en", "mbert_en", "xlmr-large_en_ud"]:
        ser_dir = "results_" + model_type 
        PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
        params = Params.from_file(PARAMS_FILE)
        Train(params, cont = True).iterate_folds()

    for model_type in ["brbert-base","brbert-large","xlmr-base", "xlmr-large", "mbert", "brbert-large_ud", "xlmr-large_ud"]:
        ser_dir = "results_" + model_type 
        PARAMS_FILE = "Configs/model_" + model_type + ".jsonnet"
        params = Params.from_file(PARAMS_FILE)
        Train(params).iterate_folds()
