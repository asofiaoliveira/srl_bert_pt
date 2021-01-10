# Portuguese SRL


This work was developed in the context of my Master's thesis in Data Science.
The code is based on [AllenNLP's package](https://github.com/allenai/allennlp) and the pre-trained models used in this work came from [🤗 Transformers](https://github.com/huggingface/transformers) and [neuralmind-ai BERTimbau - Portuguese BERT](https://github.com/neuralmind-ai/portuguese-bert).

There are two branches in this repository: one with the code used to run the experiments (`reproduce`) for the thesis and one to use for predicting (`master`) (they are only slightly different). This is due to an error in the allennlp package in the version used for the experiments.

## Models

The trained models can be obtained using the get_model.py script.

```python
python get_model.py [model name]
```

In the following table, we present all the possible model names, which model they correspond to as well as the average F<sub>1</sub> in the cross-validation <span>PropBank.Br</span> data sets and the average F<sub>1</sub> in the Buscapé set. For more information, check the article.

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

## To reproduce

To reproduce the reported results, you need first to install allennlp and allennlp_models versions 1.0.0rc3 and [iterative-stratification](https://github.com/trent-b/iterative-stratification). Note that for Windows, you'll need to install the pytorch package v1.5.0 before, with the command from their [website](https://pytorch.org). With pip:

```bash
pip install allennlp==1.0.0rc3 allennlp_models==1.0.0rc3 iterative-stratification pandas
```


Download the reproduce branch of this repository. You will also need to create a folder **data** in the same directory as the python files. In that folder, put the XML data for PropBank.Br v1.1, PropBank.Br v2 and Buscapé in a folder **xml_data**. Put the conll version of PropBank.Br in a folder **conll_data**. This data can be found [here](http://www.nilc.icmc.usp.br/portlex/index.php/en/downloads).

### Transforming XML to CoNLL data

``` python
python xml_to_conll.py
```

### Create folds

```python
python create_folds.py
```

### Run *all* the models

```python
python train.py
```

## To Predict

You need to install allennlp and allennlp_models versions 1.0.0. With pip:

```bash
pip install allennlp==1.0.0 allennlp_models==1.0.0 
```

Note that for Windows, you'll need to install the pytorch package before, with the command from their [website](https://pytorch.org).

Download the master branch of this repository. You only need to run:

```python
python my_predict.py [path/to/model] [text/to/predict]
```

The model has to be a folder (not an archive); the text to predict can be either a string or a text file.

## Citation

```
@misc{oliveira2021transformers,
      title={Transformers and Transfer Learning for Improving Portuguese Semantic Role Labeling}, 
      author={Sofia Oliveira and Daniel Loureiro and Alípio Jorge},
      year={2021},
      eprint={2101.01213},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
