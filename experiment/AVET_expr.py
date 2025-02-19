from metrics.performance.performance_metrics import perform_matching
from AVET_LFW_Generate import generate_protected_templates
from pathlib import Path 
import numpy as np
import logging
from datetime import datetime
import pandas as pd 



def log_experiment_results(logger, eer_list, optimal_thr_list, dataset,mean_time_list):
    logger.info("#" * 100)
    logger.info(f"\nFinal results for {dataset}:")
    logger.info(f"Mean EER: {np.mean(eer_list):.2f}")
    logger.info(f"Mean Optimal Threshold: {np.mean(optimal_thr_list):.2f}")
    logger.info(f"Standard Deviation of EER: {np.std(eer_list):.2f}")
    logger.info(f"Standard Deviation of Optimal Threshold: {np.std(optimal_thr_list):.2f}")
    logger.info(f"Max EER: {np.max(eer_list):.2f}")
    logger.info(f"Min EER: {np.min(eer_list):.2f}")
    logger.info(f"Max Optimal Threshold: {np.max(optimal_thr_list):.2f}")
    logger.info(f"Min Optimal Threshold: {np.min(optimal_thr_list):.2f}")
    logger.info(f"Mean time for generating protected templates: {np.mean(mean_time_list)} s, max time: {np.max(mean_time_list)} s. min time: {np.min(mean_time_list)} s.",)
    table_results.append([dataset, np.mean(eer_list)])
    logger.info("#" * 100)

if __name__ == '__main__':
    tips = "解决了matching里的迭代器bug，跑一次全流程，看看效果"
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create timestamp directory inside logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = logs_dir / f"experiment_{timestamp}_{tips}"
    experiment_dir.mkdir(exist_ok=True)

    # Configure logging to output to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(experiment_dir / f'results_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Rest of your code remains the same
    measure = "euclidean" # cosine or euclidean
    
    methods = ["baseline","bio_hash","bi_avet","avet","in_avet",] # avet or bi_avet or in_avet 
    # methods = ["bi_avet"]
    # methods = ["in_avet"]
    times = 5
    seed = np.arange(0,1000)
    datasets = {
        "fingerprint":["FVC2002/Db1_a","FVC2002/Db2_a","FVC2002/Db3_a",
                       "FVC2004/Db1_a","FVC2004/Db2_a","FVC2004/Db3_a"],
        "face":["FEI","LFW", "ColorFeret", "CASIA-WebFace"]
    }
    for method in methods:
        # Global list to store all experiment results
        table_results = []
        if method == "avet" : measure = "euclidean"
        elif method == 'bi_avet': measure = "hamming"
        elif method == 'in_avet': measure = "jaccard" 
        elif method == 'bio_hash': measure = "hamming"
        elif method == 'baseline': measure = "cosine"
        for data_type in datasets:

            if data_type == "fingerprint":
                for dataset in datasets[data_type]: 
                    Path(f"./protectedTemplates/{dataset}").mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
                    eer_list = []
                    optimal_thr_list = []
                    mean_time_list = []
                    for i in range(times):
                        mean_time = generate_protected_templates(data_type = data_type,
                                                        dataset = dataset,
                                                        seed=seed[i],
                                                        method=method
                                                        )
                        mean_time_list.append(mean_time)
                        logger.info(f"{dataset} {method} Generating 500 fingerprint protected templates costs: {mean_time:.2f} s")
                        logger.info(f"({i+1}/{times}) Matching evaluation for {dataset}... method={method} seed={seed[i]},measure={measure}")
                        EER, thr = perform_matching(data_type=data_type,
                                                    dataset=dataset,
                                                    template_path=f"./protectedTemplates/{dataset}",
                                                    verbose=True,
                                                    measure=measure)
                        eer_list.append(EER)
                        optimal_thr_list.append(thr)

                    log_experiment_results(logger, eer_list, optimal_thr_list, f"{dataset}",mean_time_list)

            elif data_type == "face":
                for dataset in datasets[data_type]:
                        Path(f"./protectedTemplates/{dataset}").mkdir(parents=True, exist_ok=True) # 确保输出目录存在
                        eer_list = []
                        optimal_thr_list = []
                        mean_time_list = []
                        for i in range(times):
                            generate_protected_templates(data_type = data_type,
                                                            dataset = dataset,
                                                            seed= seed[i],
                                                            method=method
                                                            )
                            mean_time_list.append(mean_time)
                            logger.info(f"{dataset} {method} Generating 1524 face protected templates costs ：{mean_time:.2f} s")
                            logger.info(f"({i+1}/{times}) Matching evaluation for {dataset}... method={method} seed={seed[i]},measure={measure}")
                            EER, thr = perform_matching(data_type=data_type,
                                                    dataset=dataset,
                                                    template_path=f"./protectedTemplates/{dataset}",
                                                    measure=measure)
                            eer_list.append(EER)
                            optimal_thr_list.append(thr)
                        log_experiment_results(logger, eer_list, optimal_thr_list, dataset,mean_time_list)

        # Save all experiment results at the end
        df = pd.DataFrame(table_results, columns=["Dataset", "Mean EER"])
        df.to_csv(experiment_dir / f"{method}_{measure}_seed_1~{times}_results_{timestamp}.csv", index=False)