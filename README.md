# Task Structure
Every task has to define the following:
- `Dockerfile` Setup of the docker container including all relevant preparations
- `dataset` Dataset in some kind of format
- `entities.tsv` Entities that have to be mapped to the input KG
- `README.md` README describing any deviations from the general task API, dataset format and key entities (from which graph they originate)

# Running Stuff
In the examples, we use the KG `dbpedia50k`. Adapt the identifier to run other KGs.

## Running Embedding Generation
First we need to build the docker image:
```
python . build preprocessing.embeddings
```

Then run the container:
```
python . run preprocessing.embeddings -k dbpedia50k
```

## Running all Tasks
First build the docker images (if you specify a knowledge graph, only the tasks in its config will be built):
```
python . build tasks -k dbpedia50k
```

Then run the containers:
```
python . run tasks -k dbpedia50k
```