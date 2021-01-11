import pandas as pd
import numpy as np
import os
import json
import sys
import streamlit as st
import altair as alt

from allennlp_models.structured_prediction.metrics.srl_eval_scorer import SrlEvalScorer
import matplotlib.pyplot as plt


with open("data/saved_scores.json") as f:
    data_all = json.load(f)

models=["base","large","multilingual_cased","xlmr_base","xlmr_large","xlmr_base_cont","multilingual_cased_cont","xlmr_large_cont","large_ud_cont","xlmr_large_ud_cont","xlmr_large_ud_onto_cont"]
st.title("Choosing the Best Model for Your Application")
st.sidebar.title("Parameters")
type=st.sidebar.radio("Choose the type of data you have:", ["Clean", "Unclean"])

semantic_roles = ["A0","A1","A2","A3","A4","A5","AM-ADV","AM-ASP","AM-CAU","AM-COM","AM-DIR", "AM-DIS", "AM-EXP", "AM-EXT","AM-GOL", "AM-LOC","AM-MNR","AM-MOD","AM-NEG","AM-NSE","AM-PAS","AM-PRD","AM-PRP","AM-REC","AM-TML","AM-TMP"]
not_ignore=st.sidebar.multiselect("Remove the semantic roles you do not want to use", semantic_roles, semantic_roles)

st.sidebar.header("Note")
st.sidebar.write("**Clean data** - text without or with few spelling and sentence construction errors.")
st.sidebar.write("**Unclean data** - text with spelling and sentence construction errors.")

data_to_use = data_all["test"] if type=="Clean" else data_all["buscape"]


evalu = SrlEvalScorer(ignore_classes = not_ignore+["V"])

change_names = {"base": "srl-pt_bertimbau-base",
                "large": "srl-pt_bertimbau-large",
                "xlmr_base": "srl-pt_xlmr-base",
                "xlmr_large": "srl-pt_xlmr-large",
                "multilingual_cased": "srl-pt_mbert-base",
                "xlmr_base_cont": "srl-enpt_xlmr-base",
                "xlmr_large_cont": "srl-enpt_xlmr-large",
                "multilingual_cased_cont": "srl-enpt_mbert-base",
                "large_ud_cont": "ud_srl-pt_bertimbau-base",
                "xlmr_large_ud_cont": "ud_srl-pt_xlmr-large",
                "xlmr_large_ud_onto_cont": "ud_srl-enpt_xlmr-large"}

def calc_for_model(model):
    metrics={}
    scores={}
    for fold in range(10):
        tmp = data_to_use[model][str(fold)].copy()
        scores[str(fold)]={}
        for sc in ["true_positives","false_positives","false_negatives"]:
            scores[str(fold)][sc]={}
            for tag in tmp[sc]:
                if tag in not_ignore+["V"]:
                    scores[str(fold)][sc][tag]= tmp[sc][tag]
        evalu._true_positives = scores[str(fold)]["true_positives"]
        evalu._false_positives = scores[str(fold)]["false_positives"]
        evalu._false_negatives = scores[str(fold)]["false_negatives"]
        metrics[str(fold)] = evalu.get_metric()
    
    metrics_avg = metrics["0"]

    count = dict(zip(metrics["0"].keys(), [1]*len(metrics["0"])))

    for fold in range(1,10):
        for key, value in metrics[str(fold)].items():
            metrics_avg[key] = metrics_avg.get(key,0)+value
            count[key] = count.get(key,0)+1
    
    for key, value in metrics_avg.items():
        metrics_avg[key] /= count[key]
    
    return metrics_avg

def process_metrics(data, metric = "f1-measure"):
    dic = {}
    for key, value in data.items():
        if metric in key:
            dic[key.split(metric+"-")[1]] = round(value*100,2)
    return dic

metrics2 = {}
for model in models:
    metrics2[model] = calc_for_model(model)

data=pd.DataFrame()
for model in models:
    tmp = pd.Series(process_metrics(metrics2[model]))
    tmp.name = change_names[model]
    data=data.append(tmp)

data.reset_index(inplace=True)
data = pd.melt(data, id_vars = ["index"])
data.rename(columns={'index':'Model'}, inplace=True)

m = data.iloc[np.where(data["variable"]=="overall")]["value"].argmax()
st.write("Best model: `"+data.iloc[m]["Model"]+"`\n")


selection2 = alt.selection_multi(fields=['Model'], bind='legend')
bars = alt.Chart(data.iloc[np.where(data["variable"]=="overall")]).mark_bar().encode(
    x=alt.X('Model:N',sort="-y",axis=alt.Axis(title='Model',labels =False)),
    y=alt.Y('value:Q',
        scale=alt.Scale(domain=(0,100)),axis=alt.Axis(title='')
    ), 
    color='Model:N',
    strokeWidth=alt.value(2),
    opacity=alt.condition(selection2, alt.value(1), alt.value(0.1)),
    tooltip=["Model","value"]
).add_selection(selection2)

st.write("## Overall results:")

st.altair_chart(alt.layer(bars).interactive(), use_container_width=True)



lines = alt.Chart(data.iloc[np.where(data["variable"]!="overall")]).mark_bar().encode(
    y=alt.Y('value:Q',
        scale=alt.Scale(domain=(0,100)),axis=alt.Axis(title='')
    ),
    color='Model:N',
    x=alt.X('Model:N',sort="-y",axis=alt.Axis(title='Model',labels =False)),
    tooltip=["Model","value"]
)

st.write("## Results per variable:")
st.altair_chart(lines.facet('variable:N',columns=2).resolve_scale(x='independent').interactive(), use_container_width=True)