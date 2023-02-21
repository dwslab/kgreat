def get_torchbiggraph_config():
    return dict(  # noqa
        # I/O data
        entity_path='',  # filled via script
        edge_paths=[''],  # filled via script
        checkpoint_path='',  # filled via script
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
