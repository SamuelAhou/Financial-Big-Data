import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.covariance import LedoitWolf 


class HPCAEstimator:
    def __init__(
        self, 
        k: int = 10, 
        epsilon: float = 1e-10, 
        robust: bool = False,
    ):
        """
        Initializes the HPCA estimator with sector information or clustering results.
        
        Args:
            k (int): Number of principal components to keep per cluster.
            epsilon (float): Small regularization term to stabilize covariance computations.
            robust (bool): Whether to use robust covariance estimation (Ledoit-Wolf).
        """
        self.__cov_matrix = None
        self.k = k
        self.epsilon = epsilon
        self.robust = robust


    def fit(
        self, 
        returns: pd.DataFrame, 
        clusters: Dict[str, Any],
    ) -> None:
        """
        Fits the HPCA model to asset returns data based on clusters.

        Args:
            returns (pd.DataFrame): Asset returns with assets as columns.
            clusters (Dict[str, Any]): Dictionary with tickers as keys and cluster labels as values.
        """
        n = returns.shape[1]
        clusters_list = list(clusters.keys())
        b = len(clusters_list)
        sector_eigenvalues = {}
        sector_eigenvectors = {}

        cluster_corr = self._calculate_inter_cluster_correlation(returns, clusters) # TODO: this should be the correlation between the first cluster factors

        # Compute eigenvalues and eigenvectors for each cluster
        effective_ks = []
        for cluster in clusters_list:
            cluster_returns = returns[list(clusters[cluster])]
            
            # Compute covariance matrix (robust or standard)
            if not self.robust:
                cluster_cov = np.cov(cluster_returns, rowvar=False) + self.epsilon * np.eye(cluster_returns.shape[1])
            else:
                lw = LedoitWolf()
                cluster_cov = lw.fit(cluster_returns).covariance_ + self.epsilon * np.eye(cluster_returns.shape[1])
            
            eigenvalues, eigenvectors = np.linalg.eigh(cluster_cov)
            idx = eigenvalues.argsort()[::-1]
            sector_eigenvalues[cluster] = eigenvalues[idx][:self.k]
            sector_eigenvectors[cluster] = eigenvectors[:, idx][:, :self.k]
            effective_ks.append(len(sector_eigenvalues[cluster]))

        # Adjust k to maximum effective_k if it exceeds any sector's dimensionality
        max_effective_k = min(self.k, max(effective_ks))
        if self.k > max_effective_k:
            print(f"Warning: k was set to {self.k} but the maximum allowable components is {max_effective_k}. Adjusting k to {max_effective_k}.")
            self.k = max_effective_k

        # Embed eigenvectors into the full space
        W = []
        for cluster, tickers in clusters.items():
            eigenvectors = sector_eigenvectors[cluster]
            indices = [returns.columns.get_loc(ticker) for ticker in tickers]
            effective_k = min(self.k, eigenvectors.shape[1])
            
            sector_block = []
            for i in range(effective_k):
                embedded_vector = np.zeros(n)
                embedded_vector[indices] = eigenvectors[:, i]
                sector_block.append(embedded_vector)
            W.append(sector_block)

        # Construct the M matrix
        M = np.zeros((b, b))
        for i in range(b):
            for j in range(b):
                if i == j:
                    M[i, j] = sector_eigenvalues[clusters_list[i]][0]
                else:
                    M[i, j] = (
                        np.sqrt(sector_eigenvalues[clusters_list[i]][0] *
                                sector_eigenvalues[clusters_list[j]][0]) * 
                                cluster_corr.iloc[i, j]
                    )

        # Compute eigenvalues and eigenvectors of M
        mu, alpha = np.linalg.eigh(M)
        idx = mu.argsort()[::-1]
        mu = mu[idx]
        alpha = alpha[:, idx]

        # Combine blocks in W using alpha weights to construct W_tilde
        num_global_components = min(self.k, alpha.shape[1], max(effective_ks))
        
        W_tilde = []
        for i in range(num_global_components):
            global_vector = np.zeros(n)
            for j, sector_block in enumerate(W):
                if i < len(sector_block) and i < alpha.shape[1]:
                    global_vector += alpha[j, i] * sector_block[i]
            W_tilde.append(global_vector)

        # Convert W_tilde to an (n, k) matrix for the final covariance computation
        W_tilde = np.column_stack(W_tilde)

        # Rebuild the hierarchical covariance matrix R_tilde using W_tilde and mu
        self.__cov_matrix = W_tilde @ np.diag(mu[:num_global_components]) @ W_tilde.T


    def _calculate_inter_cluster_correlation(
        self, 
        returns: pd.DataFrame, 
        clusters: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculates the inter-cluster correlation matrix.

        Args:
            returns (pd.DataFrame): DataFrame containing asset returns.
            clusters (Dict[str, list]): Dictionary of clusters with tickers.

        Returns:
            pd.DataFrame: Correlation matrix between cluster means.
        """
        cluster_means = pd.DataFrame({
            cluster: returns[tickers].mean(axis=1) for cluster, tickers in clusters.items()
        })
        return cluster_means.corr()


    def save_cov_matrix(
        self,
        filepath: str,
    ) -> None:
        """
        Saves the covariance matrix to a file.

        Args:
            filepath (str): The path where the covariance matrix will be saved.
        """
        if self.__cov_matrix is None:
            raise ValueError("Covariance matrix is not available. Please fit the model before saving.")
        np.save(filepath, self.__cov_matrix)
        print(f"Covariance matrix saved to {filepath}.")


    @property
    def cov(self) -> np.ndarray:
        """Returns the HPCA covariance matrix after fitting."""
        if self.__cov_matrix is None:
            raise ValueError("HPCA model has not been fit yet.")
        return self.__cov_matrix