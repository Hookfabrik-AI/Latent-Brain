from latent_brain.experiment import TrainConfig, evaluate, train


def test_learning_improves_heldout_transition_accuracy():
    # Small budget smoke, not a claim of general reasoning.
    config = TrainConfig(seed=42, train_maps=30, test_maps=10, epochs=6, hidden_size=32)
    brain, meta = train(config)
    report = evaluate(brain, config)
    assert meta["training_loss_by_epoch"][-1] < meta["training_loss_by_epoch"][0]
    assert report["transition_accuracy"] > report["majority_class_baseline_accuracy"]
    assert report["oracle_successes"] == report["planning_trials"]
