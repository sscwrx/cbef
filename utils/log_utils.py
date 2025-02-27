import numpy as np 

def log_experiment_results(self, logger, eer_list, optimal_thr_list, dataset, mean_time_list,dect_index_list) -> None:
    separator = "#" * 100
    logger.info(separator)
    logger.info(f"\nFinal results for {dataset}:")
    
    # EER Statistics
    mean_eer = np.mean(eer_list)
    std_eer = np.std(eer_list)
    max_eer = np.max(eer_list)
    min_eer = np.min(eer_list)
    
    # Threshold Statistics
    mean_thr = np.mean(optimal_thr_list)
    std_thr = np.std(optimal_thr_list)
    max_thr = np.max(optimal_thr_list)
    min_thr = np.min(optimal_thr_list)
    
    # Time Statistics
    mean_time = np.mean(mean_time_list)
    max_time = np.max(mean_time_list)
    min_time = np.min(mean_time_list)
    
    # DI Statistics
    mean_DI = np.mean(dect_index_list)
    max_di = np.max(dect_index_list)
    min_di = np.min(dect_index_list)


    # Output formatting
    logger.info("EER Statistics:")
    logger.info(f"  Mean: {mean_eer*100:.2f}%")
    logger.info(f"  Std Dev: {std_eer*100:.2f}%")
    logger.info(f"  Max: {max_eer*100:.2f}%")
    logger.info(f"  Min: {min_eer*100:.2f}%")
    
    logger.info("\nThreshold Statistics:")
    logger.info(f"  Mean: {mean_thr:.2f}")
    logger.info(f"  Std Dev: {std_thr:.2f}")
    logger.info(f"  Max: {max_thr:.2f}")
    logger.info(f"  Min: {min_thr:.2f}")
    
    logger.info(f"\nDecidability Index Statistics:")
    logger.info(f"  Decidability Index:")
    logger.info(f"  Mean: {mean_DI:.4f}")
    logger.info(f"  Max: {max_di:.4f}")
    logger.info(f"  Min: {min_di:.4f}")

    logger.info("\nTemplate Generation Time:")
    logger.info(f"  Mean: {mean_time*1000:.2f} ms")
    logger.info(f"  Max: {max_time*1000:.2f} ms")
    logger.info(f"  Min: {min_time*1000:.2f} ms")
    
    

    # For reference, also log the raw lists
    logger.info(f"\nRaw Data:")
    logger.info(f"  EER values: {[f'{x*100:.2f}%' for x in eer_list]}")
    logger.info(f"  Threshold values: {[f'{x:.2f}' for x in optimal_thr_list]}")
    logger.info(f"  Decidability Index values: {[f'{x:.4f}' for x in dect_index_list]}")
    logger.info(separator)
