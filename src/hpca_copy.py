import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.covariance import LedoitWolf 


class HPCAEstimator:
    def __init__(
        self, 
        k: int = 10, 
        epsilon: float = 1e-10, 
    ):
        """
        Initializes the HPCA estimator with sector information or clustering results.
        
        Args:
            k (int): Number of principal components to keep per cluster.
            epsilon (float): Small regularization term to stabilize covariance computations.
        """
        self.__cov_matrix = None
        self.k = k
        self.epsilon = epsilon


    def fit(
        self, 
        tickers: List[Tuple[str, int]],
    ) -> None:
        """
        Fits the HPCA model to asset returns data based on clusters.

        Args:
            tickers (Tuple[str]): Tuple of asset tickers (str) with their corresponding clusters (int).
        """
        n = len(tickers)
        tickers_list = [ticker for ticker, _ in tickers] # Changed
        clusters_list = list(set([cluster for _, cluster in tickers])) # Changed
        b = len(clusters_list)

        clusters = {} # Changed
        for cluster in clusters_list:
            clusters[cluster] = [ticker for ticker, c in tickers if c == cluster] 

        sector_eigenvalues = {}
        sector_eigenvectors = {}

        # Compute inter-cluster correlation matrix
        cluster_corr = self._calculate_inter_cluster_correlation(tickers, clusters)

        # Compute eigenvalues and eigenvectors for each cluster
        effective_ks = []
        for cluster in clusters_list:
            cluster_returns = [ticker for ticker, c in tickers if c == cluster] # Changed
            
            # TODO Compute covariance matrix via Hayashi-Yoshida Method
            cluster_cov = np.eye(len(cluster_returns))

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
            indices = [tickers_list.index(ticker) for ticker in tickers] # Changed
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

    
    def _upsample_returns(
        self,
        returns: np.ndarray,
        timestamps: np.ndarray,
        new_timestamps: np.ndarray,
    ) -> np.ndarray: # Changed
        """
        Upsamples the given time series to `N_samples` using linear interpolation.

        Args:
            returns (pd.Series): A pandas Series representing the log-return values.
            timestamps (pd.Series): A pandas Series of datetime objects representing the index.
            N_samples (int): The number of samples to upsample to.

        Returns:
            pd.DataFrame: A DataFrame with upsampled timestamps and interpolated returns.
        """
        # Convert timestamps to numeric values for interpolation
        old_timestamps = timestamps.astype('int64')  # Nanoseconds since epoch
        # new_timestamps = new_timestamps.astype('int64')
        
        # Interpolate returns on the new timestamp grid
        interpolated_returns = np.interp(new_timestamps, old_timestamps, returns)
        
        return interpolated_returns

    def _upsample_cluster_returns(
        self,
        cluster_tickers: List[str],
        dirData: str,
        day: int = 3,
        ) -> pd.DataFrame:  # Changed

        """
        Upsamples the assets in a cluster to the asset with the most trades.

        Args:
            cluster_tickers (List[str]): List of asset tickers in the cluster.
            dirData (str): Directory where the data is stored.
            day (int): Day of the month to consider.
        
        Returns:
            pd.DataFrame: DataFrame containing upsampled returns for all assets in the cluster.
        """

        asset_list = []
        
        # Find asset with the most trades
        most_trades = None

        for ticker in cluster_tickers:
            asset_clean = pd.read_parquet(dirData + ticker + "-trade.parquet").loc[lambda idx: idx["index"].dt.day == day]
            n_trades = len(asset_clean)
            if most_trades is None or n_trades > most_trades[1]:
                most_trades = (ticker, n_trades)
            asset_list.append(asset_clean)

        # Initialize the upsampled cluster returns with the asset with the most trades
        cluster_returns = pd.read_parquet(dirData + most_trades[0] + "-trade.parquet").loc[lambda idx: idx["index"].dt.day == day]

        N_samples = most_trades[1]
        start = cluster_returns["index"].astype('int64').min()
        end = cluster_returns["index"].astype('int64').max()
        # Generate an evenly spaced grid of N_samples timestamps
        new_timestamps = np.linspace(start, end, N_samples)
        
        cluster_returns = pd.Series(index=new_timestamps)

        for asset in asset_list:
            returns = asset["log-return"].to_numpy().astype(np.float64)
            timestamps = asset["index"]

            upsampled_returns = self._upsample_returns(returns, timestamps, new_timestamps)
            upsampled_returns = pd.Series(upsampled_returns, index=new_timestamps)

            cluster_returns = pd.concat([cluster_returns, upsampled_returns], axis=1)

        cluster_returns = cluster_returns.iloc[:, 1:]
        cluster_returns.columns = cluster_tickers

        return cluster_returns 


    def _calculate_inter_cluster_correlation(
        self, 
        tickers: List[str],
        clusters: Dict[str, Any],
    ) -> pd.DataFrame: # Changed
        """
        Calculates the inter-cluster correlation matrix.

        Args:
            returns (pd.DataFrame): DataFrame containing asset returns.
            clusters (Dict[str, list]): Dictionary of clusters with tickers.

        Returns:
            pd.DataFrame: Correlation matrix between cluster means.
        """
        dirData = "../data/clean/normal/"
        cluster_means = []

        for cluster, tickers in clusters.items():
            cluster_returns = self._upsample_cluster_returns(tickers, dirData)
            cluster_mean = cluster_returns.iloc[:, 1:].mean(axis=1)
            cluster_means.append(cluster_mean)

        # Upsample cluster means to the same length
        most_trades_cluster = cluster_means[np.argmax([len(cluster) for cluster in cluster_means])]

        start = most_trades_cluster.index[0]
        end = most_trades_cluster.index[-1]
        new_timestamps = np.linspace(start, end, len(most_trades_cluster))

        cluster_means_df = pd.Series(index=new_timestamps)

        for i, cluster in enumerate(cluster_means):
            returns = cluster.to_numpy().astype(np.float64)
            timestamps = cluster.index.to_numpy()

            upsampled_cluster_returns = self._upsample_returns(returns, timestamps, new_timestamps)
            upsampled_cluster_returns = pd.Series(upsampled_cluster_returns, index=new_timestamps)

            cluster_means_df = pd.concat([cluster_means_df, upsampled_cluster_returns], axis=1)

        cluster_means_df = cluster_means_df.iloc[:, 1:]
        cluster_means_df.columns = list(clusters.keys())

        return cluster_means_df.corr()


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
    

if __name__ == "__main__":

    import os 
    import matplotlib.pyplot as plt
    import seaborn as sns

    hpca_estimator = HPCAEstimator()

    current_dir = os.getcwd()
    tickers = np.load('tickers_sectors_filtered.npy', allow_pickle=True)

    hpca_estimator.fit(tickers)

    