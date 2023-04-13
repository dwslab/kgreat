# Task Structure
Every task has to define the following:
- `Dockerfile` Setup of the docker container including all relevant preparations
- `dataset` Dataset in some kind of format
- `entities.tsv` Entities that have to be mapped to the input KG
- `README.md` README describing any deviations from the general task API, dataset format and key entities (from which graph they originate)

# Running Stuff
In the examples, we use the task `dm-AAUP` with the KG `dbpedia50k`. Adapt the paths/identifiers to run other tasks or KGs.
All code should be executed from the project root.

## Running Embedding Generation
First build the docker container:
```
docker build -t gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat/preprocessing/embeddings -f ./shared/preprocessing/embeddings/Dockerfile .
```

Then run the docker container:
```
docker run --mount type=bind,src="$(pwd)/kg/dbpedia50k",target="/app/kg" gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat/preprocessing/embeddings
```

## Running a Task
First build the docker container (after making sure that your docker daemon is running):
```
docker build -t gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat/task/dm-aaup -f ./tasks/dm-AAUP/Dockerfile .
```

Then run the docker container:
```
docker run --mount type=bind,src="$(pwd)/kg/dbpedia50k",target="/app/kg" -e KGREAT_TASK=dm-AAUP_classification gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat/task/dm-aaup
```