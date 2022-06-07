import numpy as np
import pickle
import os
import pandas as pd

from src.utils import util_latent_analysis

def get_clustering_algorithm(model_name, data, data_iid_label, n_components, dim_red_algorithm='no_transformation'):
    if model_name == "kmeans":
        return util_latent_analysis.kmeans_fun(data=data, data_iid_label=data_iid_label, n_components =n_components, dim_red_algorithm=dim_red_algorithm)
    elif model_name == "em":
        return util_latent_analysis.em_fun(data=data,  data_iid_label=data_iid_label, n_components =n_components, dim_red_algorithm=dim_red_algorithm)
    else:
        raise ValueError(model_name)

def upload_pso_particles(path_dir, n_particles, latent_dim, patient_classes, iteration=-1, analysis='iid'):
    print(f'Upload {analysis} data from pso')
    data = np.array([], dtype='float32')
    label = np.array([], dtype='uint8')
    for pat in patient_classes:  # per label
        print(f"{analysis}_class:{pat}")
        data_pat = np.ones((n_particles, latent_dim), dtype='float32')
        with open(os.path.join(path_dir, f'particles_position_{analysis}_class_{pat}.pkl'), 'rb') as f:
            history_particles = pickle.load(f)
        for particle_idx, p_key in enumerate(history_particles.keys()):
            data_pat[particle_idx, :] = history_particles[p_key].iloc[iteration,  :]  # with -1 we select the last iteration
        if data.size == 0:
            data = data_pat.copy()
            label = np.repeat(pat, data_pat.shape[0])
        else:
            data = np.concatenate([data, data_pat], axis=0)
            label = np.concatenate([label, np.repeat(pat, data_pat.shape[0])], axis=0)
    data = pd.DataFrame(data)
    return data, label
