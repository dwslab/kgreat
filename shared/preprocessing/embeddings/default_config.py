def get_torchbiggraph_config():
    return dict(  # noqa
        # I/O data
        entity_path="./kg",
        edge_paths=[
            "./kg/embeddings",
        ],
        checkpoint_path="./kg/embeddings/model",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=200,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=50,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,
    )
