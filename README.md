
# GRM (Graph Relevance Miner) <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Friedrich-Alexander-Universit%C3%A4t_Erlangen-N%C3%BCrnberg_logo.svg/2000px-Friedrich-Alexander-Universit%C3%A4t_Erlangen-N%C3%BCrnberg_logo.svg.png" height="35">


This repository contains the implementation of GRM from [Stierle et al. (2021)](https://doi.org/10.1016/j.dss.2021.113511).
To cite this work, please use:
```
@article{stierle2020grm,
title = "A technique for determining relevance scores of process activities using graph-based neural networks",
journal = "Decision Support Systems",
pages = "113511",
volume = "144",
year = "2021",
issn = "0167-9236",
doi = "https://doi.org/10.1016/j.dss.2021.113511",
url = "http://www.sciencedirect.com/science/article/pii/S016792362100021X",
author = "Matthias Stierle and Sven Weinzierl and Maximilian Harl and Martin Matzner"
}
```
The implementation of our model is based on the GGNN implementation of [Li et al. (2015)](https://arxiv.org/abs/1511.05493) which you can find [here](https://github.com/microsoft/gated-graph-neural-network-samples).

GRM can extract the relevance scores of process activities given a performance measure from an event log.

To use GRM, you first have to preprocess your event log with the included preprocessing functions and can then use it as input data for the GRM.


## Structure of this repository
```
+-- grm
|   +-- __init__.py          		| __init file for GRM.
|   +-- GGNN.py        	    		| The base implementation of a GGNN.
|   +-- GGNNsparse.py 	    		| The sub-class of GGNN that enable the use of sparsity matrices.
|   +-- GRM.py         	    		| The sub-class of GGNNsparse that implements the Graph Relevance Miner (GRM).
|   +-- preprocessing.py       		| Script containing the preprocessing function for event logs for later use in GRM.
|   +-- util.py         	  	| Utility functions used in GRM and preprocessing.
+-- eval
|   +-- baselines    			| Dir: contains the baselines used in the evaluation.
|   |   +-- predictive_quality          | Dir: application of the baseline models (BiLSTM, RF, XGBoost) to event logs.
|   +-- case_study 			| Dir: contains the scripts used for the case study.
|   +-- data 			        | Dir: data used by GRM.
|   +-- predictive_quality 		| Dir: scripts that apply GRM to certain event logs.
|   |   +-- most_least 		   	| Dir: files from "data" with the most/least relevant activity removed.
|   +-- results		                | Dir: results of all evaluation runs and of the significance testing including the R script used
|   +-- util         			| Dir: utilities to get information from the data (characteristics) or the artefact (metrics).
+-- .gitignore 				| .gitignore file.
+-- LICENCE                       	| Licence.
+-- README.md 				| README file.
+-- environment.yml   			| environment file  .      
```

## Installation
- Install Anaconda
- `conda env create -f environment.yml` 
- `conda env update --file environment.yml  --prune`
- `conda activate grm`

To track the results of our experiments, we used [mlflow](https://github.com/mlflow/mlflow). If you want to execute any scripts in the [eval directory](eval/), you will have to [configure mlflow (with databricks)](https://databricks.com/blog/2019/10/17/managed-mlflow-now-available-on-databricks-community-edition.html) first :
- `pip install mlflow`
- `databricks configure`

## Usage
The main functions are implemented in the class GRM in [GRM.py](/grm/GRM.py). 

To use GRM with your own event log, you need to preprocess the data and then train your model before making any predictions. 
An example for the usage can be found [here](/eval/simple_test.py).

### Load event log 
Before you can use an event log with GRM, you have to preprocess the data and convert all traces in your log to Process Instance Graphs. 
For this, the method [```import_data()```](grm/preprocessing.py) is provided to load xes and csv files.

### Build model
To build the model, you have to create an instance of [```GRM```](grm/GRM.py) that receives especially the training data and the list of activities contained in the log (they can be obtained through [get_activities](grm/util.py)). 
These are needed because a subsample of the event log may not contain all distinct activities from the event log and these need to be known to the model during training.

### Restore model
Of course you can also restore models that you have already trained. 
This can be simply done by providing the path to the model file during the creation of the GRM instance with the optional argument ```restore_model```.


## Evaluate model
We did an extensive evaluation with comparisons of the GRM to BiLSTMs, XG-Boost and RandomForest.
It is important to note that our BiLSTM model is implemented with tensorflow 2.2.0 and GRM with tensorflow 1.3.0.
The data used in our implementation can be found in ```eval/data``` and utilitiy scripts that we used to get the metrics and the log characteristics in ```eval/util```.
We used the [bpi2017](https://data.4tu.nl/articles/BPI_Challenge_2017/12696884), the [bpi2018al](https://data.4tu.nl/articles/BPI_Challenge_2018/12688355), the [bpi2020pl](https://data.4tu.nl/collections/BPI_Challenge_2020/5065541) and the [sp2020](https://zenodo.org/record/3928487) (data we collected from a company) event log.

For all evaluations, we use a 10 fold cross validation and evaluated with the metrics AUC_ROC, Specificity and Sensitivity. 
