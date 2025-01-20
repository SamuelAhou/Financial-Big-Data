import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from arch import arch_model
from arch.bootstrap import StationaryBootstrap
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import root_mean_squared_error

from typing import Any, Dict, List, Optional, Tuple


def fetch_sector(
    ticker: str,
) -> Tuple[str, Optional[str]]:
    """
    Fetches the sector for a given stock ticker using the yfinance library.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        Tuple[str, Optional[str]]: A tuple containing the ticker and its sector.
    """
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get("sector")
        return ticker, sector
    except Exception:
        return ticker, None


def fetch_sectors_in_batches(
    tickers: List[str],
    batch_size: int = 10,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Fetches sector information for a list of tickers in batches using multithreading.

    Args:
        tickers (List[str]): List of stock ticker symbols.
        batch_size (int): Number of threads to use for fetching data.

    Returns:
        Tuple[Dict[str, str], List[str]]: A dictionary with ticker-sector pairs and a list of tickers with missing sector data.
    """
    sector_info: Dict[str, str] = {}
    missing_tickers: List[str] = []

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_sector, ticker): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            ticker, sector = future.result()
            if sector:
                sector_info[ticker] = sector
            else:
                missing_tickers.append(ticker)
    
    return sector_info, missing_tickers


def dict_to_labels(
    cluster_dict: Dict[int, List[str]],
    all_tickers: List[str],
) -> np.ndarray:
    """
    Convert a cluster dictionary to a 1D label array.

    Args:
        cluster_dict (Dict[int, List[str]]): Dictionary with cluster IDs as keys and lists of tickers as values.
        all_tickers (List[str]): List of all tickers.

    Returns:
        np.ndarray: Array of cluster labels corresponding to the tickers.
    """
    labels = np.empty(len(all_tickers), dtype=int)
    ticker_to_label = {
        ticker: cluster for cluster, tickers in enumerate(cluster_dict.values())
        for ticker in tickers
    }

    for idx, ticker in enumerate(all_tickers):
        labels[idx] = ticker_to_label.get(ticker, -1)

    return labels


def stationary_bootstrap_covariances(
    data: pd.DataFrame,
    p: float = 0.1,
    n_samples: int = 100,
) -> List[np.ndarray]:
    """
    Generate stationary bootstrap samples and calculate their covariance matrices.

    Args:
        data (pd.DataFrame): The input data.
        p (float): The probability of starting a new block.
        n_samples (int): The number of bootstrap samples to generate.

    Returns:
        List[np.ndarray]: List of covariance matrices from bootstrap samples.
    """
    bootstrap = StationaryBootstrap(p, data.values)
    bootstrap_cov_matrices = []

    for sample in bootstrap.bootstrap(n_samples):
        bootstrapped_cov = np.cov(sample[0][0], rowvar=False)
        bootstrap_cov_matrices.append(bootstrapped_cov)

    return bootstrap_cov_matrices


def evaluate_bootstrapped_results(
    cov_matrix_custom: np.ndarray,
    cov_matrix_sector: np.ndarray,
    bootstrapped_covariances: List[np.ndarray],
) -> None:
    """
    Compare custom and sector-based HPCA covariance matrices against bootstrapped sample covariances.

    Args:
        cov_matrix_custom (np.ndarray): Custom HPCA covariance matrix.
        cov_matrix_sector (np.ndarray): Sector HPCA covariance matrix.
        bootstrapped_covariances (List[np.ndarray]): List of bootstrapped covariance matrices.

    Returns:
        None
    """
    frobenius_distances_custom = []
    frobenius_distances_sector = []
    spectral_norm_custom = []
    spectral_norm_sector = []
    eigenvalue_distances_custom = []
    eigenvalue_distances_sector = []

    for boot_cov in bootstrapped_covariances:
        frobenius_distances_custom.append(np.linalg.norm(boot_cov - cov_matrix_custom, 'fro'))
        frobenius_distances_sector.append(np.linalg.norm(boot_cov - cov_matrix_sector, 'fro'))

        spectral_norm_custom.append(np.linalg.norm(boot_cov - cov_matrix_custom, 2))
        spectral_norm_sector.append(np.linalg.norm(boot_cov - cov_matrix_sector, 2))

        eigenvalues_boot = np.linalg.eigvalsh(boot_cov)
        eigenvalues_custom = np.linalg.eigvalsh(cov_matrix_custom)
        eigenvalues_sector = np.linalg.eigvalsh(cov_matrix_sector)

        eigenvalue_distances_custom.append(np.linalg.norm(eigenvalues_boot - eigenvalues_custom))
        eigenvalue_distances_sector.append(np.linalg.norm(eigenvalues_boot - eigenvalues_sector))

    # Calculate averages and confidence intervals for Frobenius and spectral norms, and eigenvalue distances
    avg_frobenius_custom = np.mean(frobenius_distances_custom)
    avg_frobenius_sector = np.mean(frobenius_distances_sector)
    avg_spectral_custom = np.mean(spectral_norm_custom)
    avg_spectral_sector = np.mean(spectral_norm_sector)
    avg_eigenvalue_distance_custom = np.mean(eigenvalue_distances_custom)
    avg_eigenvalue_distance_sector = np.mean(eigenvalue_distances_sector)

    ci_frobenius_custom = np.percentile(frobenius_distances_custom, [2.5, 97.5])
    ci_frobenius_sector = np.percentile(frobenius_distances_sector, [2.5, 97.5])
    ci_spectral_custom = np.percentile(spectral_norm_custom, [2.5, 97.5])
    ci_spectral_sector = np.percentile(spectral_norm_sector, [2.5, 97.5])
    ci_eigenvalue_custom = np.percentile(eigenvalue_distances_custom, [2.5, 97.5])
    ci_eigenvalue_sector = np.percentile(eigenvalue_distances_sector, [2.5, 97.5])

    print("Bootstrapped Comparison Metrics:")
    print(f"Average Frobenius Norm (Custom): {avg_frobenius_custom}")
    print(f"95% CI for Frobenius Norm (Custom): {ci_frobenius_custom}")
    print(f"Average Frobenius Norm (Sector): {avg_frobenius_sector}")
    print(f"95% CI for Frobenius Norm (Sector): {ci_frobenius_sector}")

    print(f"Average Spectral Norm (Custom): {avg_spectral_custom}")
    print(f"95% CI for Spectral Norm (Custom): {ci_spectral_custom}")
    print(f"Average Spectral Norm (Sector): {avg_spectral_sector}")
    print(f"95% CI for Spectral Norm (Sector): {ci_spectral_sector}")

    print(f"Average Eigenvalue Distance (Custom): {avg_eigenvalue_distance_custom}")
    print(f"95% CI for Eigenvalue Distance (Custom): {ci_eigenvalue_custom}")
    print(f"Average Eigenvalue Distance (Sector): {avg_eigenvalue_distance_sector}")
    print(f"95% CI for Eigenvalue Distance (Sector): {ci_eigenvalue_sector}")


def three_way_matrix_comparison(
    sample_cov: np.ndarray,
    custom_cov: np.ndarray,
    sector_cov: np.ndarray,
    bootstrap_results: bool = False,
    scaled_returns: Optional[pd.DataFrame] = None,
    p: float = 0.1,
    n_bootstrap_samples: int = 100,
) -> None:
    """
    Perform a three-way comparison of sample, custom HPCA, and sector HPCA covariance matrices.
    Optionally, use bootstrapping to compare matrices against a distribution of bootstrapped covariances.

    Args:
        sample_cov (np.ndarray): Sample covariance matrix.
        custom_cov (np.ndarray): Custom HPCA covariance matrix.
        sector_cov (np.ndarray): Sector HPCA covariance matrix.
        bootstrap_results (bool): Whether to perform bootstrapping.
        scaled_returns (Optional[pd.DataFrame]): Scaled returns data for bootstrapping.
        p (float): The probability of starting a new block for bootstrapping.
        n_bootstrap_samples (int): The number of bootstrap samples to generate.

    Returns:
        None
    """
    eigenvalues_sample = np.linalg.eigvalsh(sample_cov)
    eigenvalues_custom = np.linalg.eigvalsh(custom_cov)
    eigenvalues_sector = np.linalg.eigvalsh(sector_cov)

    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(eigenvalues_sample)[::-1], label='Sample Covariance')
    plt.plot(np.sort(eigenvalues_custom)[::-1], label='Custom Clustering Covariance')
    plt.plot(np.sort(eigenvalues_sector)[::-1], label='Sector Clustering Covariance')
    plt.title('Eigenvalue Comparison')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.xlim(0, 15)
    plt.legend()
    plt.show()

    # Distance between eigenvalues
    eigenvalue_distance_custom = np.linalg.norm(eigenvalues_sample - eigenvalues_custom)
    eigenvalue_distance_sector = np.linalg.norm(eigenvalues_sample - eigenvalues_sector)
    eigen_value_distance_cross = np.linalg.norm(eigenvalues_custom - eigenvalues_sector)
    print(f"Eigenvalue Distance (Sample vs Custom): {eigenvalue_distance_custom}")
    print(f"Eigenvalue Distance (Sample vs Sector): {eigenvalue_distance_sector}")
    print(f"Eigenvalue Distance (Custom vs Sector): {eigen_value_distance_cross}")

    cumulative_variance_sample = np.cumsum(eigenvalues_sample) / np.sum(eigenvalues_sample)
    cumulative_variance_custom = np.cumsum(eigenvalues_custom) / np.sum(eigenvalues_custom)
    cumulative_variance_sector = np.cumsum(eigenvalues_sector) / np.sum(eigenvalues_sector)

    mse_custom = root_mean_squared_error(cumulative_variance_sample, cumulative_variance_custom)**2
    mse_sector = root_mean_squared_error(cumulative_variance_sample, cumulative_variance_sector)**2
    print(f"MSE in Cumulative Explained Variance (Sample vs Custom): {mse_custom}")
    print(f"MSE in Cumulative Explained Variance (Sample vs Sector): {mse_sector}")

    eigenvectors_sample = eigh(sample_cov)[1][:, -10:]
    eigenvectors_custom = eigh(custom_cov)[1][:, -10:]
    eigenvectors_sector = eigh(sector_cov)[1][:, -10:]

    alignment_custom = np.sum(np.abs(np.dot(eigenvectors_sample.T, eigenvectors_custom))) / 10
    alignment_sector = np.sum(np.abs(np.dot(eigenvectors_sample.T, eigenvectors_sector))) / 10
    alignment_cross = np.sum(np.abs(np.dot(eigenvectors_custom.T, eigenvectors_sector))) / 10
    print(f"Eigenvector Alignment (Sample vs Custom, Top 10): {alignment_custom}")
    print(f"Eigenvector Alignment (Sample vs Sector, Top 10): {alignment_sector}")
    print(f"Eigenvector Alignment (Custom vs Sector, Top 10): {alignment_cross}")

    frobenius_norm_custom = np.linalg.norm(sample_cov - custom_cov, 'fro')
    frobenius_norm_sector = np.linalg.norm(sample_cov - sector_cov, 'fro')
    frobenius_norm_cross = np.linalg.norm(custom_cov - sector_cov, 'fro')
    spectral_norm_custom = np.linalg.norm(sample_cov - custom_cov, 2)
    spectral_norm_sector = np.linalg.norm(sample_cov - sector_cov, 2)
    spectral_norm_cross = np.linalg.norm(custom_cov - sector_cov, 2)

    print(f"Frobenius Norm (Sample vs Custom): {frobenius_norm_custom}")
    print(f"Frobenius Norm (Sample vs Sector): {frobenius_norm_sector}")
    print(f"Frobenius Norm (Custom vs Sector): {frobenius_norm_cross}")
    print(f"Spectral Norm (Sample vs Custom): {spectral_norm_custom}")
    print(f"Spectral Norm (Sample vs Sector): {spectral_norm_sector}")
    print(f"Spectral Norm (Custom vs Sector): {spectral_norm_cross}")

    upper_sample = sample_cov[np.triu_indices_from(sample_cov, k=1)]
    upper_custom = custom_cov[np.triu_indices_from(custom_cov, k=1)]
    upper_sector = sector_cov[np.triu_indices_from(sector_cov, k=1)]

    correlation_custom = np.corrcoef(upper_sample, upper_custom)[0, 1]
    correlation_sector = np.corrcoef(upper_sample, upper_sector)[0, 1]
    correlation_cross = np.corrcoef(upper_custom, upper_sector)[0, 1]
    print(f"Upper Triangular Correlation (Sample vs Custom): {correlation_custom}")
    print(f"Upper Triangular Correlation (Sample vs Sector): {correlation_sector}")
    print(f"Upper Triangular Correlation (Custom vs Sector): {correlation_cross}")

    if bootstrap_results and scaled_returns is not None:
        bootstrapped_covariances = stationary_bootstrap_covariances(scaled_returns, p, n_bootstrap_samples)
        evaluate_bootstrapped_results(custom_cov, sector_cov, bootstrapped_covariances)


def agglomerative_clustering_min_size(
    returns: pd.DataFrame,
    distance_threshold: float = 0.525,
    min_cluster_size: int = 5
) -> Dict[int, List[str]]:
    """
    Perform Agglomerative Clustering on returns using Euclidean distance, with post-processing to enforce a minimum cluster size.
    
    Args:
        returns (pd.DataFrame): Returns returns of assets with assets as columns.
        distance_threshold (float): The linkage distance threshold above which clusters are not merged.
        min_cluster_size (int): Minimum required size of each cluster.
        
    Returns:
        Dict[int, List[str]]: Dictionary where keys are cluster numbers and values are lists of tickers in each cluster.
    """
    # Compute the Euclidean distance matrix on the returns
    distance_matrix = squareform(pdist(returns.T, metric='euclidean'))
    
    agglomerative = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='complete',
        distance_threshold=distance_threshold
    )
    labels = agglomerative.fit_predict(distance_matrix)
    
    # Post-process to enforce minimum cluster size
    clusters = {}
    unique_labels = set(labels)
    
    for cluster_num in unique_labels:
        tickers = returns.columns[labels == cluster_num].tolist()
        clusters[cluster_num] = tickers

    # Identify clusters smaller than the minimum size
    small_clusters = {k: v for k, v in clusters.items() if len(v) < min_cluster_size}
    large_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

    # Merge small clusters into nearest large clusters
    for _, small_tickers in small_clusters.items():
        nearest_cluster = None
        min_distance = float('inf')
        
        # Calculate the average distance from the small cluster to each large cluster
        for large_key, large_tickers in large_clusters.items():
            small_indices = [returns.columns.get_loc(ticker) for ticker in small_tickers]
            large_indices = [returns.columns.get_loc(ticker) for ticker in large_tickers]
            
            avg_distance = np.mean([distance_matrix[si, li] for si in small_indices for li in large_indices])
            
            if avg_distance < min_distance:
                min_distance = avg_distance
                nearest_cluster = large_key

        # Merge the small cluster into the nearest large cluster
        if nearest_cluster is not None:
            large_clusters[nearest_cluster].extend(small_tickers)
    
    final_clusters = {}
    for i, tickers in enumerate(large_clusters.values(), start=1):
        final_clusters[i] = tickers
    print(final_clusters)

    assert len(set.union(*[set(v) for v in final_clusters.values()])) == len(returns.columns)

    return final_clusters


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets based on train_ratio.

    Args:
        data (pd.DataFrame): The input data to split.
        train_ratio (float): The ratio of data to use for training.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and test sets.
    """
    split_point = int(len(data) * train_ratio)
    return data.iloc[:split_point], data.iloc[split_point:]


def estimate_covariances(
    train_data: pd.DataFrame,
    custom_estimator: Any,
    sector_estimator: Any,
    custom_clusters: Dict[str, List[str]],
    sector_clusters: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Estimate covariance matrices using sample, custom HPCA, and sector HPCA.

    Args:
        train_data (pd.DataFrame): The training data.
        custom_estimator (Any): The custom HPCA estimator.
        sector_estimator (Any): The sector HPCA estimator.
        custom_clusters (Dict[str, List[str]]): Custom clusters for the assets.
        sector_clusters (Dict[str, List[str]]): Sector clusters for the assets.

    Returns:
        Dict[str, np.ndarray]: A dictionary with covariance matrices.
    """
    sample_cov = np.cov(train_data, rowvar=False)
    custom_estimator.fit(train_data, custom_clusters)
    sector_estimator.fit(train_data, sector_clusters)

    return {
        'Sample': sample_cov,
        'Custom': custom_estimator.cov,
        'Sector': sector_estimator.cov,
    }


def optimize_min_variance_portfolio(
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum-variance portfolio weights for a given covariance matrix.

    Args:
        cov_matrix (np.ndarray): The covariance matrix.

    Returns:
        np.ndarray: The portfolio weights that minimize variance.
    """
    n = cov_matrix.shape[0]
    init_guess = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]  # Long-only constraint
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] 

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    result = minimize(portfolio_variance, init_guess, bounds=bounds, constraints=constraints)
    return result.x


def calculate_realized_variance(
    test_data: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Calculate the realized portfolio variance over the test period.

    Args:
        test_data (pd.DataFrame): The test data.
        weights (np.ndarray): The portfolio weights.

    Returns:
        float: The realized variance of the portfolio.
    """
    portfolio_returns = test_data.dot(weights)
    return np.var(portfolio_returns)


def rolling_windows(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate rolling windows of train and test sets.

    Args:
        data (pd.DataFrame): The input data.
        train_size (int): The size of the training set.
        test_size (int): The size of the test set.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of tuples containing train and test sets.
    """
    windows = []
    start = 0
    while start + train_size + test_size <= len(data):
        train_data = data.iloc[start:start + train_size]
        test_data = data.iloc[start + train_size:start + train_size + test_size]
        windows.append((train_data, test_data))
        start += test_size 
    return windows


def out_of_sample_risk_prediction(
    data: pd.DataFrame,
    custom_estimator: Any,
    sector_estimator: Any,
    custom_clusters: Dict[str, List[str]],
    sector_clusters: Dict[str, List[str]],
    train_size: int = 252,
    test_size: int = 21,
) -> Dict[str, float]:
    """
    Perform out-of-sample risk prediction using sample, custom HPCA, and sector HPCA estimators.

    Args:
        data (pd.DataFrame): The input data.
        custom_estimator (Any): The custom HPCA estimator.
        sector_estimator (Any): The sector HPCA estimator.
        custom_clusters (Dict[str, List[str]]): Custom clusters for the assets.
        sector_clusters (Dict[str, List[str]]): Sector clusters for the assets.
        train_size (int): The size of the training set.
        test_size (int): The size of the test set.

    Returns:
        Dict[str, float]: A dictionary with the average realized variances for each estimator.
    """
    windows = rolling_windows(data, train_size, test_size)
    realized_variances = {'Sample': [], 'Custom': [], 'Sector': []}

    for train_data, test_data in windows:
        covariances = estimate_covariances(train_data,
                                           custom_estimator,
                                           sector_estimator,
                                           custom_clusters,
                                           sector_clusters)

        # Calculate minimum-variance portfolio weights for each covariance matrix
        weights = {
            key: optimize_min_variance_portfolio(cov_matrix)
            for key, cov_matrix in covariances.items()
        }

        # Calculate realized out-of-sample variance for each portfolio
        for key, weight_vec in weights.items():
            realized_variance = calculate_realized_variance(test_data, weight_vec)
            realized_variances[key].append(realized_variance)

    # Calculate the average out-of-sample variance for each estimator
    avg_realized_variances = {
        key: np.mean(variances) for key, variances in realized_variances.items()
    }

    return avg_realized_variances


def fit_garch_model(
    asset_returns: pd.Series,
    asset_name: str,
    model_type: str,
    max_p: int,
    max_q: int,
    criterion: str,
    verbose: bool = False,
) -> Tuple[str, pd.Series]:
    """
    Fit a GARCH model for a single asset's returns, selecting the optimal (p, q) order.
    
    Args:
        asset_returns (pd.Series): Series of returns for a single asset.
        asset_name (str): Name of the asset.
        model_type (str): Type of GARCH model to use ('GARCH', 'EGARCH', 'GJR-GARCH').
        max_p (int): Maximum lag order for GARCH term.
        max_q (int): Maximum lag order for ARCH term.
        criterion (str): Information criterion for model selection ('AIC' or 'BIC').
        verbose (bool): Whether to print model fitting information.
    
    Returns:
        tuple: Asset name, best order (p, q), and standardized residuals (de-GARCHed returns).
    """
    best_criterion = np.inf
    best_order = (1, 1)
    asset_returns = asset_returns * 100 # Scale returns for numerical stability
    
    # Grid search for best (p, q) order based on AIC or BIC
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                if model_type == 'GARCH':
                    model = arch_model(asset_returns, vol='Garch', p=p, q=q)
                elif model_type == 'EGARCH':
                    model = arch_model(asset_returns, vol='EGarch', p=p, q=q)
                elif model_type == 'GJR-GARCH':
                    model = arch_model(asset_returns, vol='Garch', p=p, o=1, q=q)
                else:
                    raise ValueError("Invalid model_type. Choose 'GARCH', 'EGARCH', or 'GJR-GARCH'.")
                
                fit = model.fit(disp='off')
                model_criterion = fit.aic if criterion == 'AIC' else fit.bic
                
                if model_criterion < best_criterion:
                    best_criterion = model_criterion
                    best_order = (p, q)
            except Exception as e:
                print(f"Could not fit {model_type}({p}, {q}) for {asset_name}: {e}")

    # Fit the final model with the optimal order
    best_p, best_q = best_order
    final_model = arch_model(asset_returns, vol=model_type.lower(), p=best_p, q=best_q)
    final_fit = final_model.fit(disp='off')
    standardized_residuals = final_fit.resid / final_fit.conditional_volatility
    standardized_residuals = standardized_residuals / 100  # Rescale residuals back to original units
    
    if verbose:
        print(f"Asset: {asset_name}, Best Order: {model_type}({best_p}, {best_q}), {criterion}: {best_criterion}")
    
    return asset_name, standardized_residuals


def de_garch_returns_parallel(
    returns: pd.DataFrame,
    model_type: str = 'GARCH',
    max_p: int = 3,
    max_q: int = 3,
    criterion: str = 'AIC'
) -> pd.DataFrame:
    """
    Apply GARCH de-volatilization to a DataFrame of returns by selecting the optimal model 
    order based on an information criterion, using parallel processing.

    Args:
        returns (pd.DataFrame): DataFrame of asset returns.
        model_type (str): Type of GARCH model to use. Options are 'GARCH', 'EGARCH', or 'GJR-GARCH'.
        max_p (int): Maximum lag order for the GARCH term to consider.
        max_q (int): Maximum lag order for the ARCH term to consider.
        criterion (str): Information criterion for model selection ('AIC' or 'BIC').

    Returns:
        pd.DataFrame: DataFrame of de-GARCHed returns.
    """
    de_garched_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(fit_garch_model, returns[asset], asset, model_type, max_p, max_q, criterion): asset
            for asset in returns.columns
        }

        for future in as_completed(futures):
            asset_name, standardized_residuals = future.result()
            de_garched_returns[asset_name] = standardized_residuals

    return de_garched_returns