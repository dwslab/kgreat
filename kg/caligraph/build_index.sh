mkdir -p ./tdb
mkdir -p ./index

podman run \
  -p 9273:9273 \
  -e MAVEN_OPTS='-Xmx700G -Xms10G' \
  -v ./index_config.yml:/root/app-config.yml \
  -v ./data:/root/data/ \
  -v ./tdb:/root/tdb/ \
  -v ./index:/root/index/ \
  dbpedia/dbpedia-lookup

