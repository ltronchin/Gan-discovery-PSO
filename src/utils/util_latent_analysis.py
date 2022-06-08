import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import umap
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm
from scipy import linalg
import matplotlib as mpl
import itertools
import pandas as pd

### Dimesionality reduction analysis
def pca_fun(plot_training_dir, data, data_iid_label, iteration):
    print('Run PCA')
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)

    # Variance plot
    exp_var_perc = [np.sum(pca.explained_variance_[:i]) for i in range(len(pca.explained_variance_))] / np.sum(pca.explained_variance_)
    fig = plt.figure(figsize=[8, 6])
    plt.xlabel('PCA component')
    plt.ylabel('Explained variance')
    plt.plot(exp_var_perc, linestyle='-', linewidth=2.0)
    fig.savefig(os.path.join(plot_training_dir, f"pca_variance_plot_{iteration}.png"), dpi=400, format='png')
    #plt.show()

    data_pca = pca.transform(data)  # Applichiamo i "loads" della PCA ai dati in ingresso proiettandoli sui nuovi assi
    fig = plt.figure(figsize=[8, 6])
    for y in np.unique(data_iid_label):
        c_map = data_iid_label == y
        plt.scatter(data_pca[c_map, 0], data_pca[c_map, 1], s=50, cmap='Spectral', marker='.', label=str(y))
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('PC1', fontsize=20)
    plt.ylabel('PC2', fontsize=20)
    plt.title('PCA projection of latent space (iid class)', fontsize=20)
    plt.legend()
    fig.savefig(os.path.join(plot_training_dir, f"pca_space_{iteration}.png"), dpi=400, format='png')
    #plt.show()

    return pca

def umap_fun(plot_training_dir, data, data_iid_label, iteration):
    print('Run UMAP')
    reducer = umap.UMAP(random_state=42)
    reducer.fit(data)

    data_umap = reducer.transform(data)

    fig = plt.figure(figsize=[8, 6])
    for y in np.unique(data_iid_label):
        c_map = data_iid_label == y
        plt.scatter(data_umap[c_map, 0], data_umap[c_map, 1], s=50, marker='.', cmap='spectral', label=str(y))
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP1', fontsize=20)
    plt.ylabel('UMAP2', fontsize=20)
    plt.title('UMAP projection of latent space (iid class)', fontsize=20)
    plt.legend()
    fig.savefig(os.path.join(plot_training_dir, f"umap_space_{iteration}.png"), dpi=400, format='png')
    #plt.show()

    return reducer

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

def voronoi_diagram(plot_training_dir, reduced_data_iid, data_iid_label, centroids, dim_red_algorithm, reduced_data_ood=None, data_ood_label=None, colorise=None):
    # Compute Voronoi tesselation
    points = centroids.copy()
    vor = Voronoi(points)
    fig = plt.figure(figsize=[8, 6])
    if colorise is not None:
        regions, vertices = voronoi_finite_polygons_2d(vor)
        # Colorize
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.4)
        plt.plot(points[:, 0], points[:, 1], 'ko')
        plt.xlim(vor.min_bound[0] - 2, vor.max_bound[0] + 2)
        plt.ylim(vor.min_bound[1] - 2, vor.max_bound[1] + 2)
    else:
        voronoi_plot_2d(vor, show_points=True, show_vertices=False, line_colors='black', line_width=2, line_alpha=0.6,   point_size=12)
        for y in np.unique(data_iid_label):
            c_map = data_iid_label == y
            plt.scatter(reduced_data_iid[c_map, 0], reduced_data_iid[c_map, 1], s=20, marker='.', cmap='spectral', label=str(y))
        plt.legend()

        if reduced_data_ood is not None:
            plt.scatter(reduced_data_ood[:1000, 0], reduced_data_ood[:1000, 1], s=1, marker='x', edgecolors=None, color='black', alpha=1)

    plt.title(f"Voronoi diagram of the latent space")
    plt.xlabel(f'{dim_red_algorithm}_1')
    plt.ylabel(f'{dim_red_algorithm}_2')
    if colorise is not None:
        plt.savefig(os.path.join(plot_training_dir, f"voronoi_colour_{dim_red_algorithm}.png"), dpi=400, format='png')
    else:
        if reduced_data_ood is not None:
            plt.savefig(os.path.join(plot_training_dir, f"voronoi_ood_{data_ood_label}_{dim_red_algorithm}.png"), dpi=400, format='png')
        else:
            plt.savefig(os.path.join(plot_training_dir, f"voronoi_{dim_red_algorithm}.png"), dpi=400, format='png')
    plt.show()

def plot_latent_space(plot_training_dir, data_iid, data_iid_label, centroids=None, data_ood=None, data_ood_label=None,dim_reduction_algorithm=None):

    if dim_reduction_algorithm is not None:
        fig_title= f"{dim_reduction_algorithm} Latent Space"
        fig_xlabel = f"{dim_reduction_algorithm}_1"
        fig_ylabel = f"{dim_reduction_algorithm}_2"
        fig_savename = f"latent_space_{dim_reduction_algorithm}"
    else:
        fig_title = "Latent Space"
        fig_xlabel ='Z_1'
        fig_ylabel = 'Z_2'
        fig_savename = "latent_space"

    if data_ood is not None:
        fig_savename += f"_ood_{data_ood_label}"

    fig = plt.figure(figsize=[8, 6])
    if centroids is not None:
        plt.plot(centroids[:, 0], centroids[:, 1], 'ko')
    for y in np.unique(data_iid_label):
        c_map = data_iid_label == y
        plt.scatter(data_iid[c_map, 0], data_iid[c_map, 1], s=40, marker='o',  label=str(y))
    plt.legend()

    if data_ood is not None:
        plt.scatter(data_ood[:1000, 0], data_ood[:1000, 1], s=10, marker='x', edgecolors=None, color='black', alpha=0.3)

    plt.title(fig_title)
    plt.xlabel(fig_xlabel)
    plt.ylabel(fig_ylabel)
    fig.savefig(os.path.join(plot_training_dir, fig_savename+".png"), dpi=400,  format='png')
    fig.show()

def plot_ellipsoids(plot_training_dir, X, Y_, means, covariances, covariance_type='tied', dim_red_algorithm=None): # todo check plot for workstation

    if dim_red_algorithm is not None:
        fig_title= f"{dim_red_algorithm} Gaussian Mixture"
        fig_xlabel = f"{dim_red_algorithm}_1"
        fig_ylabel = f"{dim_red_algorithm}_2"
        fig_savename = f"ellipsoid_{dim_red_algorithm}"
    else:
        fig_title = "Latent Space"
        fig_xlabel ='Z_1'
        fig_ylabel = 'Z_2'
        fig_savename = "ellipsoid"

    fig = plt.figure(figsize=[8, 6])
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    color_iter = itertools.cycle(
        ["navy", "c", "cornflowerblue", "gold", "darkorange", "darkviolet", "forestgreen", "salmon", "lightcoral",
         "deepskyblue"])
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):

        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(fig_title)
    plt.xlabel(fig_xlabel)
    plt.ylabel(fig_ylabel)
    fig.savefig(os.path.join(plot_training_dir, fig_savename+".png"), dpi=400,  format='png')
    fig.show()

def kmeans_fun(data, data_iid_label, n_components = 2, dim_red_algorithm='pca'):
    if dim_red_algorithm == 'pca':
        # Dim reduction
        pca = PCA(n_components=n_components)
        pca.fit(data)
        reduced_data = pca.transform(data)
        # Clustering
        kmeans = KMeans(init="k-means++", n_clusters=len(np.unique(data_iid_label)), n_init=10, random_state=42)
        kmeans.fit(reduced_data.astype('double'))
        return kmeans, pca, reduced_data
    elif dim_red_algorithm == 'umap':
        # Dim reduction
        reducer =  umap.UMAP(n_components=n_components, random_state=42)
        reducer.fit(data)
        reduced_data = reducer.transform(data)
        # Clustering
        kmeans = KMeans(init="k-means++", n_clusters=len(np.unique(data_iid_label)), n_init=10, random_state=42)
        kmeans.fit(reduced_data.astype('double'))
        return kmeans, reducer, reduced_data
    elif dim_red_algorithm == 'no_transformation':
        # Clustering
        kmeans = KMeans(init="k-means++", n_clusters=len(np.unique(data_iid_label)), n_init=10, random_state=42)
        kmeans.fit(data.astype('double'))
        return kmeans
    else:
        raise ValueError(dim_red_algorithm)

def get_initial_means(X, n_components, init_params, random_state):     # Run a GaussianMixture with max_iter=0 to output the initalization means
    gmm = GaussianMixture(n_components=n_components, init_params=init_params, tol=1e-9, max_iter=1, random_state=random_state).fit(X)
    return gmm.means_

def get_initial_means_supervised(estimator, X, Y):
    estimator.means_init = np.array(
        [X[Y == i].mean(axis=0) for i in np.unique(Y)]
    )

def em_fun(data, data_iid_label, n_components = 2, dim_red_algorithm='pca'):
    if dim_red_algorithm == 'pca':
        # Dim reduction
        pca = PCA(n_components=n_components)
        pca.fit(data)
        reduced_data = pca.transform(data)
        # Clustering
        gmm = GaussianMixture(n_components=len(np.unique(data_iid_label)), max_iter=2000, random_state=42)
        gmm.means_init = get_initial_means_supervised(estimator=gmm, X=reduced_data, Y=data_iid_label)
        gmm.fit(reduced_data.astype('double'))
        return gmm, pca, reduced_data

    elif dim_red_algorithm == 'umap':
        # Dim reduction
        reducer =  umap.UMAP(n_components=n_components, random_state=42)
        reducer.fit(data)
        reduced_data = reducer.transform(data)
        # Clustering
        gmm = GaussianMixture(n_components=len(np.unique(data_iid_label)), max_iter=2000, random_state=42)
        gmm.means_init = get_initial_means_supervised(estimator=gmm, X=reduced_data, Y=data_iid_label)
        gmm.fit(reduced_data.astype('double'))
        return gmm, reducer, reduced_data

    elif dim_red_algorithm == 'no_transformation':
        gmm = GaussianMixture(n_components=len(np.unique(data_iid_label)), max_iter=2000, random_state=42)
        gmm.means_init = get_initial_means_supervised(estimator=gmm, X=data, Y=data_iid_label) #  gmm.means_init = get_initial_means(X=data.astype('double'), n_components=len(np.unique(data_iid_label)), init_params="k-means++", random_state=42)
        gmm.fit(data.astype('double'))
        return gmm
    else:
        raise ValueError(dim_red_algorithm)

def kmeans_predict(data, kmeans, reducer, dim_reduction=None):
    # dim reduction
    if dim_reduction is not None:
        data = reducer.transform(data)
    # clustering
    pred = kmeans.predict(data.astype('double'))
    return pred

def perturb_and_plot():
    pass

def pairwise_distance():
    pass

def mutual_distance(X1, X2):

    step = 0
    n = X1.shape[0] * X2.shape[0] # todo here
    mse = np.empty((n, 1), dtype=np.dtype('float32'))
    with tqdm(total=n) as pbar:
        for step_1, x1 in enumerate(X1.to_numpy()): #todo check step iterator
            for step_2, x2 in enumerate(X2.to_numpy()):
                mse[step] = np.linalg.norm(x1 - x2)
                step += 1
                pbar.set_description(f'Latent vector ood_class_1 {step_1}, latent vector ood_class_2 {step_2}')
                pbar.update(1)
    return mse



