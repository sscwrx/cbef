from experiment.base_experiment import ExperimentConfig 
from data.face_dataset import FaceDatasetConfig 
from experiment.base_experiment import ExperimentConfig ,Experiment
from config.method_configs import METHOD_CONFIGS
if __name__ == "__main__":

    dataset_config = FaceDatasetConfig(dataset_name="CASIA-WebFace") 
    method_config,verifier_config = METHOD_CONFIGS["C_IOM"]

    expr_config = ExperimentConfig(
        expriment_name="testCIOM",
        method_config=method_config,
        dataset_config=dataset_config,
        verifier_config=verifier_config,
        expriment_times=5
    )

    experiment:Experiment = expr_config.setup()

    experiment.run()