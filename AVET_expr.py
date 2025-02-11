from AVET_LFW_Matching import perform_matching
from AVET_LFW_Generate import generate_protected_templates
from pathlib import Path 
import numpy as np
import logging
from datetime import datetime
import pandas as pd 



def log_experiment_results(logger, eer_list, optimal_thr_list, dataset):
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
    table_results.append([dataset, np.mean(eer_list)])
    logger.info("#" * 100)

if __name__ == '__main__':

    # Configure logging to output to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(f'chaos_IoM_results_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Rest of your code remains the same
    measure = "euclidean" # cosine or euclidean
    
    methods = ["baseline","bio_hash","in_avet","bi_avet","avet"] # avet or bi_avet or in_avet 
    # methods = ["in_avet"]
    times = 5
    seed = [1,2,3,4,5]
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
        elif method == 'baseline': measure = "euclidean"
        for data_type in datasets:

            if data_type == "fingerprint":
                for dataset in datasets[data_type]: 
                    Path(f"./protectedTemplates/{dataset}").mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
                    eer_list = []
                    optimal_thr_list = []
                    for i in range(times):
                        mean_time = generate_protected_templates(data_type = data_type,
                                                        dataset = dataset,
                                                        seed=seed[i],
                                                        method=method
                                                        )
                        logger.info(f"{dataset} {method} 生成500个受保护模板的平均时间是：{mean_time}")
                        logger.info(f"({i+1}/5) Matching evaluation for {dataset}... method={method} seed={seed[i]},measure={measure}")
                        EER, thr = perform_matching(data_type=data_type,
                                                    dataset=dataset,
                                                    template_path=f"./protectedTemplates/{dataset}",
                                                    measure=measure)
                        eer_list.append(EER)
                        optimal_thr_list.append(thr)

                    log_experiment_results(logger, eer_list, optimal_thr_list, f"{dataset}")

            elif data_type == "face":
                for dataset in datasets[data_type]:
                        Path(f"./protectedTemplates/{dataset}").mkdir(parents=True, exist_ok=True) # 确保输出目录存在
                        eer_list = []
                        optimal_thr_list = []
                        for i in range(times):
                            generate_protected_templates(data_type = data_type,
                                                            dataset = dataset,
                                                            seed= seed[i],
                                                            method=method
                                                            )
                            logging.info(f"{dataset} {method} 生成1524个受保护模板的平均时间是：{mean_time}")
                            logger.info(f"({i+1}/5) Matching evaluation for {dataset}... method={method} seed={seed[i]},measure={measure}")
                            EER, thr = perform_matching(data_type=data_type,
                                                    dataset=dataset,
                                                    template_path=f"./protectedTemplates/{dataset}",
                                                    measure=measure)
                            eer_list.append(EER)
                            optimal_thr_list.append(thr)
                        log_experiment_results(logger, eer_list, optimal_thr_list, dataset)

        # Save all experiment results at the end
        df = pd.DataFrame(table_results, columns=["Dataset", "Mean EER"])
        df.to_csv(f"{method}_{measure}_seed_{seed}results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv", index=False)