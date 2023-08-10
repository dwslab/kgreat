# KGrEaT: **K**nowledge **Gr**aph **E**v**a**luation via Downstream **T**asks
KGrEaT is a framework built to evaluate the performance impact of knowledge graphs (KGs) on multiple downstream tasks.
To that end, the framework implements various algorithms to solve tasks like classification, regression, or recommendation of entities.
The impact of a given KG is measured by using its information as background knowledge for solving the tasks.
To compare the performance of different KGs on downstream tasks, a fixed experimental setup with the KG as the only variable is used.

## Prerequisites
### Hardware Requirements
The hardware requirements of the framework are dominated by the embedding generation step (see [DGL-KE](https://github.com/awslabs/dgl-ke) framework for details).
To compute embeddings for KGs with the size of DBpedia or YAGO, we recommend to use a CPU and have at least 100GB of RAM.
As of now, the datasets are moderate in size and the implemented algorithms are quite efficient.
Hence, the execution of tasks does not consume a large amount of resources.


### Software Requirements
- Environment manager: [conda](https://docs.continuum.io/anaconda/install/)
- Dependency manager: [poetry](https://python-poetry.org/docs/#installation)
- Container manager: [Docker](https://www.docker.com)

### Setup
- In the project root, create a conda environment with: `conda env create -f environment.yaml`
- Activate the new environment with `conda activate kgreat`
- Install dependencies with `poetry install`
- Make sure that the `kgreat` environment is activated when using the framework!


## Quickstart: How to Evaluate a KG
### KG Setup
- Create a new folder under `kg` which will contain all data related to the graph (input files, configuration, intermediate representations, results, logs). Note that the name of the folder will serve as identifier for the graph throughout the framework.
- In the folder of your KG:
  - Create a sub-folder `data`. Put the RDF files of the KG in this folder (supported file types are NT, TTL, TSV). You may want to create a download script similar to the existing KGs.
  - Create a file `config.yaml` with the evaluation configuration of your KG. You can find explanations for all configuration parameters in the `example_config.yaml` file of the root directory.
  
### Evaluation Pipeline
In the following you will prepare and run the three stages `Mapping`, `Preprocessing`, and `Task`. As the later stages are dependent on the earlier ones, they must be run in this order.

First, pull the docker images of all stages. Make sure that your `config.yaml` is already configured correctly, as the manager only pulls images of the steps defined in the config. In the root directory of the project, run the following commands:
```shell
python . <your-kg-identifier> pull
```

We then run the `prepare` action which initializes required files for the actual stages. In particular, we create a `entity_mapping.tsv` file which contains all the URIs and labels of entities to be mapped. 
```shell
python . <your-kg-identifier> prepare
```

Then we run the actual stages:
```shell
python . <your-kg-identifier> run
```

### Running Individual Stages or Steps
If you want to trigger individual stages or steps, you can do so by supplying them as optional arguments. You can trigger steps by supplying the ID of the step as defined in the `config.yaml`. Here are some examples:

Running only the preprocessing stage:
```shell
python . <your-kg-identifier> run --stage preprocessing
```

Running the RDF2vec embedding generation step of the preprocessing stage:
```shell
python . <your-kg-identifier> run --stage preprocessing --step embedding-rdf2vec
```

Running two specific classification tasks (i.e., steps of the `Task` stage):
```shell
python . <your-kg-identifier> run --stage task --step dm-aaup_classification dm-cities_classification
```

### Results & Analysis
The results of the evaluation runs are put in a `result` folder within your KG directory. The framework creates one TSV result file and one log file per task.
You can use the `result_analysis.ipynb` notebook to explore and compare the results of one or more KGs.


## How to Extend the Framework
Contributions to the framework are highly welcome, and we appreciate pull requests
for additional datasets, tasks, matchers, preprocessors, etc.! Here's how you can extend the framework:

### Add a Dataset
To add a dataset for an existing task type, create a folder in the `dataset` directory with at least the following data:
- `Dockerfile` Setup of the docker container including all relevant preparations (import code, install dependencies, ..).
- `dataset` Dataset in a format of your choice. Have a look at `shared/dm/utils/dataset.py` for already supported dataset formats
- `entities.tsv` Labels and URIs of the dataset entities that have to be mapped to the input KG
- `README.md` A file describing the dataset as well as any deviations from the general task API

To run a task using the new dataset you have to add an entry in your `config.yaml` file where you define an identifier as well as necessary parameters for your task. Don't forget to update the `example_config.yaml` with information about the new dataset/task!

### Add a Task Type
To define a new task type, add the code to a subfolder below `shared`. If your task type uses Python, you can put it below `shared/dm` and reuse the utility functions in `shared/dm/util`.
The only information a task retrieves is the environment variable `KGREAT_STEP` which it can use to identify its configuration in the `config.yaml` of the KG.
Results should be written in the `result/run_<run_id>` folder of the KG using the existing format. 

### Add a Mapper
To define a new mapper, add the code to a subfolder below `shared/mapping`. The mapper should be self-contained and should define its own `Dockerfile` (see existing mappers for examples).
A mapper should fill gaps in the `source` column of the `entity_mapping.tsv` file in the KG folder (i.e., load the file, fill gaps, update the file).

To use the mapper, add a respective entry to the mapping section of your `config.yaml`.

### Add a Preprocessing Method
To define a new preprocessing method, add the code to a subfolder below `shared/preprocessing`. The preprocessing method should be self-contained and should define its own `Dockerfile` (see existing preprocessors for examples).
A preprocessing step can use any data contained in the KG folder and persist artifacts in the same folder. These artifacts may then be used by subsequent preprocessing steps or by tasks.

To use the preprocessing method, add a respective entry to the preprocessing section of your `config.yaml`.