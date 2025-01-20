##############################################################################
# This script contains the functions used to wrangle the data for the project.
##############################################################################

# basic imports
import pandas as pd
import polars as pl
import numpy as np
import os
import glob
import tqdm
# imports needed for deGARCH
from arch import arch_model
from concurrent.futures import as_completed, ProcessPoolExecutor
# imports needed for Hayashi-Yoshida
from sklearn.covariance import LedoitWolf, EmpiricalCovariance



#######################################
###### preliminary data wrangling #####
#######################################

def flag_bad_folders(
        tickers: list[str], 
        raw_path: str
    ) -> None:
    """
    This function takes a list of tickers and renames the folders to 'bad-{ticker}' 
    if they contain files flagged as bad (*.parquet-bad).
    
    ARGS:
    -----
        tickers: 
            List of ticker names.
        raw_path: 
            Path to the directory containing the ticker folders.

    Returns:
    --------
        None
    """
    for ticker in tickers:
        # Get the path to the folder
        folder_path = os.path.join(raw_path, ticker)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"{ticker} folder does not exist.")
            continue
        
        # Get the list of files in the folder
        files = os.listdir(folder_path)
        
        # Check if the folder contains any bad flagged files
        if any(file.endswith('.parquet-bad') for file in files):
            bad_folder_path = os.path.join(raw_path, f"bad-{ticker}")
            # Rename the folder
            os.rename(folder_path, bad_folder_path)
            print(f"Folder '{ticker}' renamed to 'bad-{ticker}'.")


def wrangle_trade_file(
        DF,
        tz_exchange="America/New_York",
        only_non_special_trades=True,
        only_regular_trading_hours=True,
        merge_sub_trades=True,
        log_return=True,
    ) -> pl.LazyFrame:
    """
    Wrangle the raw trade data in DF.
    
    Args:
    -----
        DF: 
            polars DataFrame containing raw trade data.
        tz_exchange: 
            Timezone of the exchange.
        only_non_special_trades: 
            Flag to filter out special trades.
        only_regular_trading_hours: 
            Flag to filter out trades outside regular trading hours.
        merge_sub_trades: 
            Flag to merge sub-trades.
            
    Returns:
    --------
        DF:
            polars DataFrame containing wrangled trade data.
    """
    
    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30

    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    
    if log_return:
        DF = DF.with_columns([pl.col("trade-price").cast(pl.Float32),
                              (pl.col("trade-price")/pl.col("trade-price").shift(1)).cast(pl.Float32).alias("return"),
                              ])   
        DF = DF.with_columns(pl.col("return").log().alias("log-return"))
    
    if only_non_special_trades:
        DF=DF.filter(pl.col("trade-stringflag")=="uncategorized")

    # Drop Unnecessary Columns
    DF = DF.drop(["xltime","trade-rawflag","trade-stringflag", "return"])

    if merge_sub_trades:   # average volume-weighted trade price here
        DF=DF.group_by('index',maintain_order=True).agg([#(pl.col('trade-price')*pl.col('trade-volume')).sum()/(pl.col('trade-volume').sum()).alias('trade-price'),
                                                         pl.col('log-return').sum().alias('log-return')])        

    return DF


def clean_trade_file(
        filename,
        tz_exchange="America/New_York",
        only_non_special_trades=True,
        only_regular_trading_hours=True,
        merge_sub_trades=True,
    ) -> pl.LazyFrame:
    """
    Load and Wrangles raw trade files in filename. 
    If given multiple files, it will concatenate them.

    Args:
    -----
        filename: 
            Path to the trade file.
        tz_exchange: 
            Timezone of the exchange.
        only_non_special_trades: 
            Flag to filter out special trades.
        only_regular_trading_hours: 
            Flag to filter out trades outside regular trading hours.
        merge_sub_trades: 
            Flag to merge sub-trades.
    
    Returns:
    --------
        DF: 
            polars DataFrame containing wrangled trade data.
    """
    try:
        if filename.endswith("csv") or filename.endswith("csv.gz"):
            DF=pl.scan_csv(filename)
        elif filename.endswith("parquet"):    
            DF=pl.scan_parquet(filename)
        else:
            print("cannot load file "+filename+" : unknown format")
            return None
    except:
        print(filename+" cannot be loaded")
        return None

    DF = wrangle_trade_file(DF,
            tz_exchange=tz_exchange,
            only_non_special_trades=only_non_special_trades,
            only_regular_trading_hours=only_regular_trading_hours,
            merge_sub_trades=merge_sub_trades,)

    return DF


def merge_trade_files(
        ticker_name: str = None,
        raw_path: str = None,
        result_path: str = None,
        show_progress: bool = False,
    ) -> None:
    """
    Merge all the trade files for a given ticker in raw_path and save the result in result_path.

    Args:
    -----
        ticker_name: 
            Ticker name.
        raw_path: 
            Path to the directory containing the raw trade files.
        result_path: 
            Path to save the merged trade files.
        show_progress: 
            Flag to show progress.

    Returns:
    --------
        None
    """

    if ticker_name is None or raw_path is None or result_path is None:
        raise ValueError("Please provide a ticker name, raw data path, and result path.")
    
    filename = raw_path+ticker_name+"/*parquet"
    # Load all the trade files
    try:
        if filename.endswith("csv") or filename.endswith("csv.gz"):
            DF=pl.scan_csv(filename)
        elif filename.endswith("parquet"):    
            DF=pl.scan_parquet(filename)
        else:
            print("cannot load file "+filename+" : unknown format")
            return None
    except:
        print(filename+" cannot be loaded")
        return None
    
    DF.collect(streaming=True).write_parquet(result_path+"/"+ticker_name+"-trade.parquet")

    if show_progress:
        print("Merged trade files for "+ticker_name)
    
    return None


def clean_and_save_trade_files(
        tickers: list[str],
        raw_path: str,
        result_path: str,
        show_progress: bool = False,
        tz_exchange="America/New_York",
        only_non_special_trades=True,
        only_regular_trading_hours=True,
        merge_sub_trades=True,
    ) -> None:
    """
    Clean and save all the trade files for a list of tickers in raw_path and save the result in result_path.

    Args:
    -----
        tickers: 
            List of ticker names.
        raw_path: 
            Path to the directory containing the raw trade files.
        result_path: 
            Path to save the cleaned trade files.
        show_progress: 
            Flag to show progress.
        tz_exchange: 
            Timezone of the exchange.
        only_non_special_trades: 
            Flag to filter out special trades.
        only_regular_trading_hours: 
            Flag to filter out trades outside regular trading hours.
        merge_sub_trades: 
            Flag to merge sub-trades.

    Returns:
    --------
        None
    """
    for ticker in tqdm.tqdm(tickers):
        filename = raw_path+ticker+"-trade.parquet"
        DF = clean_trade_file(
                filename,
                tz_exchange=tz_exchange,
                only_non_special_trades=only_non_special_trades,
                only_regular_trading_hours=only_regular_trading_hours,
                merge_sub_trades=merge_sub_trades,
            )
        DF.collect(streaming=True).write_parquet(result_path+"/"+ticker+"-trade.parquet")


def check_number_obs_per_day(
        directory_path: str, 
        day: str = '2010-05-03', 
        show_progress: bool = False,
    ) -> list:
    """
    Check the number of observations for each ticker for a specific day.

    Args:
    -----
        directory_path: 
            Path to the directory containing the trade files.
        day: 
            Specific day to check the number of observations.
        show_progress: 
            Flag to show progress.
    
    Returns:
    --------
        obs_per_ticker: 
            List containing the number of observations for each ticker.
    """
    tickers = os.listdir(directory_path)
    
    tickers = [ticker.split("-")[0] if "-" in str(ticker) else ticker for ticker in tickers]
    if ".DS_Store" in tickers:
        tickers.remove('.DS_Store')
    if "deGARCH" in tickers:
        tickers.remove('deGARCH')
    if "SPY-trade.parquet" in tickers:
        tickers.remove('SPY-trade.parquet')
    
    obs_per_ticker = [None] * len(tickers)

    for idx, ticker in tqdm.tqdm(enumerate(tickers)):
        df = pd.read_parquet(os.path.join(directory_path, f"{ticker}-trade.parquet"))
        filtered_df = df[df['index'].dt.date == pd.to_datetime(day).date()]
        obs_per_ticker[idx] = filtered_df.shape[0]
        
        if show_progress:
            print(f"Number of observations for {ticker} for the specific date: ", obs_per_ticker[idx])
    
    return obs_per_ticker


def filter_tickers_by_obs(
        directory_path: str, 
        target_directory_path: str,
        day: str = '2010-05-03', 
        min_obs: int = 1000, 
        show_progress: bool = False,
    ) -> None:
    """
    Filter tickers based on the minimum number of observations for a specific day and
        save the filtered trade files in the target directory.

    Args:
    -----
        directory_path: 
            Path to the directory containing the trade files.
        target_directory_path: 
            Path to save the filtered trade files.
        day: 
            Specific day to check the number of observations.
        min_obs: 
            Minimum number of observations.
        show_progress: 
            Flag to show progress.
    
    Returns:
    --------
        None
    """

    tickers = os.listdir(directory_path)
    
    tickers = [ticker.split("-")[0] if "-" in str(ticker) else ticker for ticker in tickers]
    if ".DS_Store" in tickers:
        tickers.remove('.DS_Store')
    if "deGARCH" in tickers:
        tickers.remove('deGARCH')
    if "SPY-trade.parquet" in tickers:
        tickers.remove('SPY-trade.parquet')
    
    kept = 0

    for ticker in tqdm.tqdm(tickers):
        df = pd.read_parquet(os.path.join(directory_path, f"{ticker}-trade.parquet"))

        if df[df['index'].dt.date == pd.to_datetime(day).date()].shape[0] >= min_obs:
            df = df[df['index'].dt.date == pd.to_datetime(day).date()]
            df.to_parquet(os.path.join(target_directory_path, f"{ticker}-trade.parquet"))
            kept += 1
        else:
            if show_progress:
                print(f"{ticker} has less than {min_obs} observations and will not be saved.")
        
        if show_progress:
            print(f"Number of observations for {ticker}: ", df.shape[0])

    print(f"Number of tickers kept: {kept}")    
    
    return None


#############################
########## deGARCH ##########
#############################


def de_garch_returns(
    asset_name: str,
    max_p: int=1,
    max_q: int=1,
    criterion: str='BIC',
    stop_n_rows: int | None = None,
    verbose: bool = False,
    scaling: int = 10**4,
    directory_path: str = "../data/clean/filtered/",
    target_directory_path: str = "../data/clean/deGARCH/",
) -> None:
    """
    Fit a GARCH model for a single asset's returns, selecting the optimal (p, q) order. It then 
    transforms the returns to standardized residuals and saves the result to a new file.
    
    Args:
    -----
        asset_returns (pd.Series): 
            Series of returns for a single asset.
        asset_name (str): 
            Name of the asset.
        max_p (int): 
            Maximum lag order for GARCH term.
        max_q (int): 
            Maximum lag order for ARCH term.
        criterion (str): 
            Information criterion for model selection ('AIC' or 'BIC').
        verbose (bool): 
            Whether to print model fitting information.
        scaling (int):
            Scaling factor for numerical stability.
        directory_path (str):
            Path to the directory containing the asset returns.
        target_directory_path (str):
            Path to save the de-GARCHed returns.
    
    Returns:
    --------
        None
    """

    # Load the asset returns
    asset = pl.read_parquet(f"{directory_path}{asset_name}-trade.parquet", n_rows=stop_n_rows).to_pandas()
    asset_returns = asset["log-return"]

    best_criterion = np.inf
    best_order = (1, 1)

    asset_returns = asset_returns * scaling # Scale returns for numerical stability
    
    # Grid search for best (p, q) order based on AIC or BIC
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try: 
                model = arch_model(asset_returns, vol='Garch', p=p, q=q)
                fit = model.fit(disp='off')
                model_criterion = fit.aic if criterion == 'AIC' else fit.bic
                
                if model_criterion < best_criterion:
                    best_criterion = model_criterion
                    best_order = (p, q)
            except Exception as e:
                print(f"Could not fit ({p}, {q}) for {asset_name}: {e}")

    # Fit the final model with the optimal order
    best_p, best_q = best_order
    final_model = arch_model(asset_returns, vol='Garch', p=best_p, q=best_q, dist='t')
    final_fit = final_model.fit(disp='off')
    standardized_residuals = final_fit.resid / final_fit.conditional_volatility
    standardized_residuals = standardized_residuals / scaling  # Rescale residuals back to original units
    
    if verbose:
        print(f"Asset: {asset_name}, Best Order: ({best_p}, {best_q}), {criterion}: {best_criterion}")
    

    asset["log-return"] = standardized_residuals

    asset.to_parquet(f"{target_directory_path}{asset_name}-trade.parquet", engine='pyarrow')

    return None


def de_garch_returns_parallel(
    tickers: list[str],
    max_p: int = 3,
    max_q: int = 3,
    criterion: str = 'BIC',
    stop_n_rows: int | None = None,
    verbose: bool = False,
    scaling: int = 10**4,
    directory_path: str = "../data/clean/filtered/",
    target_directory_path: str = "../data/clean/deGARCH/",
    n_workers: int = os.cpu_count(),
) -> None:
    """
    Apply GARCH de-volatilization to a list of tickers using parallel processing.

    Args:
        tickers (list[str]): 
            List of asset tickers to process.
        max_p (int): 
            Maximum lag order for GARCH term.
        max_q (int):  
            Maximum lag order for ARCH term.
        criterion (str): 
            Information criterion for model selection ('AIC' or 'BIC').
        stop_n_rows (int | None): 
            Number of rows to load from the input file.
        verbose (bool): 
            Whether to print detailed progress.
        scaling (int): 
            Scaling factor for numerical stability.
        directory_path (str): 
            Path to the input data directory.
        target_directory_path (str): 
            Path to the output data directory.
        n_workers (int): 
            Number of parallel processes to use.

    Returns:
        None
    """
    # Ensure the target directory exists
    if not os.path.exists(target_directory_path):
        os.makedirs(target_directory_path)

    # Process assets in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                de_garch_returns,
                ticker,
                max_p,
                max_q,
                criterion,
                stop_n_rows,
                verbose,
                scaling,
                directory_path,
                target_directory_path,
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
                if verbose:
                    print(f"Successfully processed {ticker}.")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    print("All assets processed.")


#######################################
########## Hayashi-Yoshida ############
#######################################

def hayashi_yoshida_high_frequency(
        asset1_name, 
        asset2_name,
        scale = 1, # Rescale returns for numerical stability
        directory_path: str = "data/cleaned/deGARCH/"):
    """
    Efficiently computes the Hayashi-Yoshida covariance estimator for two time series with asynchronous timestamps.
    
    Parameters:
        asset1 (str): First time series with timestamps as the index.
        asset2 (str): Second time series with timestamps as the index.
        
    Returns:
        float: The Hayashi-Yoshida covariance estimate.
    """
    optimize = False
    stop_at_rows = 10_000

    if asset1_name == asset2_name:
        return pd.read_parquet(f"{directory_path}{asset1_name}-trade.parquet").iloc[:,1].to_numpy().var()#, n_rows= stop_at_rows)


    # Ensure time series are sorted by index
    if optimize:
        asset1 = pl.read_parquet(f"{directory_path}{asset1_name}-trade.parquet", n_rows= stop_at_rows).sort("index").to_pandas().set_index("index")
        asset2 = pl.read_parquet(f"{directory_path}{asset2_name}-trade.parquet", n_rows= stop_at_rows).sort("index").to_pandas().set_index("index")
    else:
        asset1 = pd.read_parquet(f"{directory_path}{asset1_name}-trade.parquet").set_index("index")#, n_rows= stop_at_rows)
        asset2 = pd.read_parquet(f"{directory_path}{asset2_name}-trade.parquet").set_index("index")#, n_rows= stop_at_rows)

    # Create a DataFrame by combining timestamps and values from both series
    df1 = asset1.rename(columns={'log-return': 'value1'})
    df1["flag"] = 1
    df2 = asset2.rename(columns={'log-return': 'value2'})
    df2["flag"] = 2
    merged = pd.concat([df1, df2], axis=0).sort_index()

    merged['value1'] = merged['value1'] * scale
    merged['value2'] = merged['value2'] * scale

    # Compute forward differences for each series (fill missing values with 0)
    merged['value1'] = merged['value1'].ffill()
    merged['value2'] = merged['value2'].ffill()
    merged['delta1'] = merged['value1'].diff().fillna(0)
    merged['delta2'] = merged['value2'].diff().fillna(0)

    # Drop rows where both deltas are zero (no contribution to covariance)
    merged = merged[(merged['delta1'] != 0) | (merged['delta2'] != 0)]

    # Compute the covariance contribution for overlapping intervals
    merged['cov_contrib'] = merged['delta1'] * merged['delta2']
    hy_cov = merged.loc[(merged['flag'] == 1) | (merged['flag'] == 2), 'cov_contrib'].sum()

    return hy_cov

def correlation_from_covariance(covariance):
    """ Compute the correlation matrix from the covariance matrix. """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def estimate_hayashida_yoshida_cov(data: np.ndarray, already_centered: bool = True) -> np.ndarray:
    """ Estimate the covariance matrix using the Hayashida-Joshida method.
    
    Args:
        data (np.ndarray): The input data. Assumed shape is (n_timepoints, n_assets).
        
    Returns:
        np.ndarray: The estimated covariance matrix. Shape is (n_assets, n_assets).
    """

    cov_est = EmpiricalCovariance(assume_centered=already_centered)
    cov_est.fit(data)

    return cov_est.covariance_

def estimate_ledoit_wolf_cov(data: np.ndarray, already_centered: bool = True) -> np.ndarray:
    """ Estimate the covariance matrix using the Ledoit-Wolf method.
    
    Args:
        data (np.ndarray): The input data. Assumed shape is (n_timepoints, n_assets).
        
    Returns:
        np.ndarray: The estimated covariance matrix. Shape is (n_assets, n_assets).
    """
    cov_est = LedoitWolf(assume_centered=already_centered)
    cov_est.fit(data)

    return cov_est.covariance_
