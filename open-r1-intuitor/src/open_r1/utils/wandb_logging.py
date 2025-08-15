import os


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group


def init_mlflow_training(training_args):
    """
    Helper function for setting up MLflow logging tools.
    """
    # if training_args.mlflow_experiment_id is not None:
    #     os.environ["MLFLOW_EXPERIMENT_ID"] = training_args.mlflow_experiment_id
    # if training_args.mlflow_tracking_uri is not None:
    #     os.environ["MLFLOW_TRACKING_URI"] = training_args.mlflow_tracking_uri
    # if training_args.mlflow_registry_uri is not None:
    #     os.environ["MLFLOW_REGISTRY_URI"] = training_args.mlflow_registry_uri
    pass