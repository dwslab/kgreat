# KGrEaT: **K**nowledge **Gr**aph **E**v**a**luation via Downstream **T**asks
KGrEaT is a framework built to evaluate the performance impact of knowledge graphs (KGs) on multiple downstream tasks.
To that end, the framework implements various algorithms to solve tasks like classification, regression, or recommendation of entities.
The impact of a given KG is measured by using its information as background knowledge for solving the tasks.
To compare the performance of different KGs on downstream tasks, a fixed experimental setup with the KG as the only variable is used.

## Quickstart: How to Evaluate a KG
### KG Setup
- Create a new folder under `kg` which will contain all data related to the graph (input files, configuration, intermediate representations, results, logs). Note that the name of the folder will serve as identifier for the graph throughout the framework.
- In the folder of your KG:
  - Create a sub-folder `data`. Put the RDF files of the KG in this folder (supported file types are NT, TTL, TSV). You may want to create a download script similar to the existing KGs.
  - Create a file `config.yaml` with the evaluation configuration of your KG. You can find explanations for all configuration parameters in the `example_config.yaml` file of the root directory.
  
### Evaluation Pipeline
In the following you will run steps of the three stages `Preprocessing`, `Mapping`, and `Tasks`. The first two stages have no dependencies among each other and can be run in parallel.

First, pull the docker images of the stages. Make sure that your `config.yaml` is already configured correctly, as the manager only pulls images of the steps defined in the config. In the root directory of the project, run the following commands:
```shell
python . <your-kg-identifier> pull preprocessing/embedding
python . <your-kg-identifier> pull preprocessing/ann
python . <your-kg-identifier> pull mapping
python . <your-kg-identifier> pull tasks
```

To run the mapping, we first prepare a `entity_mapping.tsv` file which contains all the URIs and labels of entities to be mapped. Then, we run the actual mapping to find the corresponding entities in your KG. For the prepare step, it is important that the images of the tasks have already been pulled as the entities are collected directly from the images. 
```shell
python . <your-kg-identifier> prepare mapping
python . <your-kg-identifier> run mapping
```

The preprocessing can be run with the following commands:
```shell
python . <your-kg-identifier> run preprocessing/embedding
python . <your-kg-identifier> run preprocessing/ann  # optional step for speed-up
```

When the mapping and preprocessing stages are completed, the actual tasks can be run.
```shell
python . <your-kg-identifier> run tasks
```

### Results & Analysis
The results of the evaluation runs are put in a `result` folder within your KG directory. The framework creates one TSV result file and one log file per task.
You can use the `result_analysis.ipynb` notebook to explore and compare the results of one or more KGs.


## How to Extend the Framework
Contributions to the framework are highly welcome and we would appreciate pull requests
for additional datasets, tasks, matchers, preprocessors, etc.! Here's how you can extend the framework:

### Add a Dataset
To add a dataset for an existing task type, create a folder in the `tasks` directory with at least the following data:
- `Dockerfile` Setup of the docker container including all relevant preparations (import code, install dependencies, ..).
- `dataset` Dataset in a format of your choice. Have a look at `shared/dm/utils/dataset.py` for already supported dataset formats
- `entities.tsv` Labels and URIs of the dataset entities that have to be mapped to the input KG
- `README.md` A file describing the dataset as well as any deviations from the general task API

To run a task using the new dataset you have to add an entry in your `config.yaml` file where you define an identifier as well as necessary parameters for your task. Don't forget to update the `example_config.yaml` with information about the new dataset/task!

### Add a Task Type
To define a new task type, add the code to a subfolder below `shared`. If your task type uses Python, you can put it below `shared/dm` and reuse the utility functions in `shared/dm/util`.

### Add a Mapper
To define a new mapper, add the code to a subfolder below `shared/mapping`. The mapper should be self-contained and should define its own `Dockerfile` (see existing mappers for examples). To use the mapper, add a respective entry to the mapping section of your `config.yaml`.

### Add a Preprocessing Method
To define a new preprocessing method, add the code to a subfolder below `shared/preprocessing`. The preprocessing method should be self-contained and should define its own `Dockerfile` (see existing preprocessors for examples). To use the preprocessing method, add a respective entry to the preprocessing section of your `config.yaml`.

