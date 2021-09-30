import glob
import numpy as np
import pandas as pd

if __name__ == '__main__':
    all_results = {'EKF': {'EKS': [0, 0],
                           'EM': [0, 0],
                           'TME-2': [0, 0],
                           'TME-3': [0, 0]},
                   'EM': {'EKS': [0, 0],
                          'EM': [0, 0],
                          'TME-2': [0, 0],
                          'TME-3': [0, 0]},
                   'TME-2': {'EKS': [0, 0],
                             'EM': [0, 0],
                             'TME-2': [0, 0],
                             'TME-3': [0, 0]},
                   'TME-3': {'EKS': [0, 0],
                             'EM': [0, 0],
                             'TME-2': [0, 0],
                             'TME-3': [0, 0]}
                   }
    for filter_name in all_results.keys():
        for smoother_name in all_results[filter_name].keys():
            file_name = glob.glob(f'../triton/results/{filter_name}_{smoother_name}_*.npy')[0]
            data = np.load(file_name)
            data = data[data <= 100]  # Remove divergent runs (only happened once for EKF-EKS)
            m, std = np.mean(data), np.std(data)
            all_results[filter_name][smoother_name] = (m, std)

    table = pd.DataFrame.from_dict(all_results)
    table.T.to_csv('rmse_table.csv')
