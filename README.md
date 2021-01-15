# Semantic Role Labeling in Portuguese: Improving the State of the Art with Transfer Learning and BERT-based Models

This work was developed in the context of my Master's thesis in Data Science.
The code is based on [AllenNLP's package](https://github.com/allenai/allennlp) and the pre-trained models used came from [🤗 Transformers](https://github.com/huggingface/transformers) and [neuralmind-ai BERTimbau - Portuguese BERT](https://github.com/neuralmind-ai/portuguese-bert).

There are three branches in this repository, which correspond to three different versions of the AllenNLP package. The branch [`v1.0.0rc3`](https://github.com/asofiaoliveira/srl_bert_pt/tree/v1.0.0rc3) contains the code used to train the models reported in the article. The branch [`v1.0.0`](https://github.com/asofiaoliveira/srl_bert_pt/tree/v1.0.0) contains the code used to test the models reported. The models were trained and tested in different versions because of a bug in version 1.0.0rc3 of AllenNLP which prevented testing some models. The main branch contains the code needed to make predictions with the trained models.

## Models

The trained models can be obtained using the get_model.py script in the main branch.

```bash
python get_model.py [model name]
```

This is necessary because the trained SRL model is split into two; the [transformers portion of the model](https://huggingface.co/liaad) is stored in the [🤗 Transformers community models](https://huggingface.co) and the linear layer is stored in this repository (in the folder [Models](Models/)). The `get_model.py` file joins these two portions and saves the complete model in a folder `[model name]`. 

In the following table, we present all the possible model names, a small description of the model, the average F<sub>1</sub> in the cross-validation <span>PropBank.Br</span> data sets and the average F<sub>1</sub> in the Buscapé set. For more information, please refer to the article.

| Model Name | F<sub>1</sub> CV PropBank.Br (in domain) | F<sub>1</sub> Buscapé (out of domain) | Explanation |
| --------------- | ------ | ----- | ------- |
| `srl-pt_bertimbau-base` | 76.30 | 73.33 | The (monolingual) BERTimbau<sub>base</sub> model trained on Portuguese SRL data |
| `srl-pt_bertimbau-large` | 77.42 | 74.85 | The (monolingual) BERTimbau<sub>large</sub> model trained on Portuguese SRL data |
| `srl-pt_xlmr-base` | 75.22 | 72.82 | The (multilingual) XLM-R<sub>base</sub> model trained on Portuguese SRL data |
| `srl-pt_xlmr-large` | 77.59 | 73.84 | The (multilingual) XLM-R<sub>large</sub> model trained on Portuguese SRL data |
| `srl-pt_mbert-base` | 72.76 | 66.89 | The multilingual cased BERT model trained on Portuguese SRL data |
| `srl-en_xlmr-base` | 66.59 | 65.24 | The (multilingual) XLM-R<sub>base</sub> model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and tested on Portuguese SRL data |
| `srl-en_xlmr-large` | 67.60 | 64.94 | The (multilingual) XLM-R<sub>large</sub> model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and tested on Portuguese SRL data |
| `srl-en_mbert-base` | 63.07 | 58.56 | The multilingual cased BERT model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and tested on Portuguese SRL data |
| `srl-enpt_xlmr-base` | 76.50 | 73.74 | The (multilingual) XLM-R<sub>base</sub> model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and then on Portuguese SRL data |
| `srl-enpt_xlmr-large` | **78.22** | 74.55 | The (multilingual) XLM-R<sub>large</sub> model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and then on Portuguese SRL data |
| `srl-enpt_mbert-base` | 74.88 | 69.19 | The multilingual cased BERT model trained on English SRL data (specifically a pre-processed CoNLL-2012 data set) and then on Portuguese SRL data |
| `ud_srl-pt_bertimbau-large` | 77.53 | 74.49 | The (monolingual) BERTimbau<sub>large</sub> model trained first in dependency parsing with the Universal Dependecies Portuguese data set and then on Portuguese SRL data |
| `ud_srl-pt_xlmr-large` | 77.69 | 74.91 | The (monolingual) XLM-R<sub>large</sub> model trained first in dependency parsing with the Universal Dependecies Portuguese data set and then on Portuguese SRL data |
| `ud_srl-enpt_xlmr-large` | 77.97 | **75.05** | The (monolingual) XLM-R<sub>large</sub> model trained first in dependency parsing with the Universal Dependecies Portuguese data set, then on English SRL data (specifically a pre-processed CoNLL-2012 data set) and finally on Portuguese SRL data |


## To Predict

In order to use the trained models for SRL prediction, first install alennlp and allennlp_models v1.2.2. With pip:

```bash
pip install allennlp==1.2.2 allennlp_models==1.2.2
```

Download the main branch of this repository. From the list of available models (see Table above), choose the one most indicated for your application (see [Choosing the best model](#choosing-the-best-model)) below for help choosing) and download the model using:

```bash
python get_model.py [model name]
```

Then run the `my_predict.py` script with the chosen model and the text you want to predict SRL labels for.

```bash
python my_predict.py [model name] [text/to/predict]
```

`[text/to/predict]` can be either a string or the path to a text file containing the text you want to predict SRL labels for.

### Example

```bash
python get_model.py srl-pt_bertimbau-large

python my_predict.py srl-pt_bertimbau-large "Só precisa ganhar experiência"
#or
python my_predict.py srl-pt_bertimbau-large pred.txt #where pred.txt contains Só precisa ganhar experiência
```

## Choosing the best model

We provide an implementation of the heuristic mentioned in the article, described by the following figure (taken from the article mentioned in [Citation](#citation)). 

![Image of heuristic](/Choose%20Best%20Model/decision_diagram_white.png)

To run the `Choose Best Model/tool.py` script, you must install streamlit.

```bash
pip install streamlit

streamlit run "Choose Best Model/tool.py"
```

In this app, you can choose the semantic roles of interest for your application (by removing the ones that do not interest you) and the type of data you have. The results will be the best model and plots showing the total F<sub>1</sub> measure and the F<sub>1</sub> measure for each role achieved by each model.


## Branch v1.0.0rc3

To reproduce the results, it is first necessary to train the models. For that, first install the pytorch package v1.5.0 with the command from their [website](https://pytorch.org/get-started/previous-versions/) according to the CUDA version of your machine, and then allennlp, allennlp_models, [iterative-stratification](https://github.com/trent-b/iterative-stratification) and pandas.

```bash
pip install allennlp==1.0.0rc3 allennlp_models==1.0.0rc3 iterative-stratification pandas
```

Next, clone or download the `v1.0.0rc3` branch of this repository. 

The data must be manually added. The code expects there to be a `data` folder (inside the folder with this repository). Within this folder, there must be 4 folders:
* `xml_data` -- contains the XML data for PropBank.Br v1.1, PropBank.Br v2 and Buscapé. This data can be found [here](http://www.nilc.icmc.usp.br/portlex/index.php/en/downloads).
* `conll_data` -- contains the conll version of PropBank.Br. This data can be found [here](http://www.nilc.icmc.usp.br/portlex/index.php/en/downloads).
* `ud` -- contains the [Portuguese Universal Depdencies](https://github.com/UniversalDependencies/UD_Portuguese-Bosque/tree/master) dataset.
* `conll-formatted-ontonotes-5.0-12` -- contains the conll formatted OntoNotes v5.0.

### Transforming XML to CoNLL data

``` bash
python xml_to_conll.py
```

### Create folds

```bash
python create_folds.py
```

### Train *all* the models

```bash
python train.py
```


## Branch v1.0.0

To reproduce the results, it is then necessary to test the models. For that, first install the pytorch package v1.6.0 with the command from their [website](https://pytorch.org/get-started/previous-versions/) according to the CUDA version of your machine, and then allennlp, allennlp_models, [iterative-stratification](https://github.com/trent-b/iterative-stratification) and pandas.

```bash
pip install allennlp==1.0.0 allennlp_models==1.0.0 iterative-stratification pandas
```

Next, clone or download the `v1.0.0` branch of this repository. 

The data must be manually added -- simply copy the `data` folder obtained previously.

### Test *all* the models

```bash
python train.py
```

Besides the metrics for each test fold and Buscapé, the program also outputs for each tested pair (model,dataset) a file with the predicted and gold tags.

## Citation

```bibtex
@misc{oliveira2021transformers,
      title={Transformers and Transfer Learning for Improving Portuguese Semantic Role Labeling}, 
      author={Sofia Oliveira and Daniel Loureiro and Alípio Jorge},
      year={2021},
      eprint={2101.01213},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
