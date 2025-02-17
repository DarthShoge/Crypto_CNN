import json
import os
import pandas as pd
import numpy as np
import dotenv
import asyncio
import pickle
from datetime import datetime
from tqdm.asyncio import trange

from image_explorer import display_image
from image_generation import generate_price_image
from coingecko_provider import CoinGeckoAPI, get_single_asset_data
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # For tqdm progress bar with multiprocessing

dotenv.load_dotenv()


def map_cg_assets(asset_names, mapping_file_path):
    # Read the JSON mapping from the file
    with open(mapping_file_path, 'r') as file:
        mapping_dict = json.load(file)

    return {v:k for k,v in mapping_dict.items() if v in asset_names } 


def extract_asset_names(directory):
    asset_names = []
    # List all files in the given directory
    for file_name in os.listdir(directory):
        # Check if the file name follows the expected pattern
        if file_name.endswith("USD_1.csv"):
            # Extract the asset name from the file name
            asset_name = file_name.split("USD_1.csv")[0]
            asset_names.append(asset_name)
    return asset_names



def process_single_asset(ticker, ohlc_data, mcaps):
    # Resampling differently for each column
    hourly_df = pd.DataFrame()
    hourly_df['Open'] = ohlc_data['Open'].resample('H').first()  # First value for Open
    hourly_df['High'] = ohlc_data['High'].resample('H').max()    # Maximum value for High
    hourly_df['Low'] = ohlc_data['Low'].resample('H').min()      # Minimum value for Low
    hourly_df['Close'] = ohlc_data['Close'].resample('H').last() # Last value for Close
    hourly_df['Volume'] = ohlc_data['Volume'].resample('H').sum() # Sum for Volume
    hourly_df['Trades'] = ohlc_data['Trades'].resample('H').sum() # Sum for Trades

    # Resetting the index to turn 'Timestamp' back into a column
    hourly_df.reset_index(inplace=True)
    # Forward fill for price data
    hourly_df[['Open', 'High', 'Low', 'Close']] = hourly_df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
    # Replace NaNs with 0 for Volume and Trades
    hourly_df[['Volume', 'Trades']] = hourly_df[['Volume', 'Trades']].fillna(0)
    hourly_df['Timestamp'] = pd.to_datetime(hourly_df['Timestamp'])
    hourly_df['Timestamp'] = pd.to_datetime(hourly_df['Timestamp'])
    hourly_df.set_index('Timestamp', inplace=True)
    # Assuming the daily data column is named 'bitcoin'
    merged_df = pd.merge_asof(hourly_df, mcaps, left_index=True, right_index=True)
    # Reset index if you want 'Timestamp' back as a column
    merged_df.reset_index(inplace=True)
    merged_df['asset'] = ticker
    
     # Calculate the 20-period moving average of the close
    merged_df['ma'] = merged_df['Close'].rolling(window=20).mean()
    return merged_df
    
    
async def process_all_assets(asset_list, cg_map, cg_api):
    assets = []
    for ticker in asset_list:
        if ticker not in cg_map:
            continue
        cg_name = cg_map[ticker]
        processed_asset = await process_single_file(cg_map, cg_api, ticker)
        assets.append(processed_asset)
    return pd.concat(assets)

async def process_single_file(cg_map, cg_api, ticker):
    price,mcaps,_ = await get_single_asset_data(cg_map[ticker], cg_api)
    ohlc_data = pd.read_csv(f'./raw_data/{ticker}USD_1.csv', header=None)
    ohlc_data.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", "Trades"]
    ohlc_data['Timestamp'] = pd.to_datetime(ohlc_data['Timestamp'], unit='s')
    ohlc_data.set_index('Timestamp', inplace=True)
    processed_asset = process_single_asset(ticker, ohlc_data, mcaps)
    return processed_asset


def process_label_and_images_df(ticker, asset_df, offset, window, short_period, full_period=24, image_size=(64, 48), from_time='09:00:00'):

    result_df = pd.DataFrame(columns=[
    'Asset',
    'Start_Date', 
    'Predict_Date',
    'End_Date', 
    'Daily_Return', 
    f'Ret_{full_period}H', 
    f'Ret_{short_period}H', 
    f'Log_Ret_{full_period}H',
    f'Log_Ret_{short_period}H',
    'Market_Cap', 
    'Image'])

    # Calculate the index of the first window
    time_offset = asset_df[asset_df['Timestamp'].dt.time == pd.to_datetime(from_time).time()].index.min()

    for i in range(offset+time_offset, len(asset_df) - (full_period + window), full_period):
        try:
            start_date = asset_df.iloc[i]['Timestamp']
            predict_date = asset_df.iloc[i + window]['Timestamp']
            # end_date = asset_df.iloc[i + window]['Timestamp']
            end_date = asset_df.iloc[i + window + full_period]['Timestamp']

            # Get the current block of data
            window_data = asset_df.iloc[i:i + window]

            # Generate the image
            image = generate_price_image(window_data, image_size)
        
            # Calculate returns
            start_price = asset_df.iloc[i + window]['Close']
            market_cap = asset_df.iloc[i + window]['mcap']
            s_period_end_price = asset_df.iloc[i + window + short_period]['Close']
            f_period_end_price = asset_df.iloc[i + window + full_period]['Close']
            
            short_period_log_return = np.log(s_period_end_price / start_price)
            long_period_log_return = np.log(s_period_end_price / start_price)
            
            short_period_return = (s_period_end_price - start_price) / start_price
            full_period_return = (f_period_end_price - start_price) / start_price

            # Store the results in the new DataFrame
            result_df = result_df._append({
            'Asset': ticker, 
            'Start_Date': start_date, 
            'Predict_Date': predict_date, 
            'End_Date': end_date, 
            f'Ret_{full_period}H': full_period_return, 
            f'Ret_{short_period}H': short_period_return, 
            f'Log_Ret_{full_period}H': long_period_log_return,
            f'Log_Ret_{short_period}H': short_period_log_return,
            'Market_Cap': market_cap,	
            'Image': image
        }, ignore_index=True)
        except Exception as e:
            print(f"Error processing {ticker} at index {i}: {e}")
    return result_df

def process_asset_parallel(args):
    """
    Wrapper function for processing a single asset.
    This function will be called in parallel for different tickers.
    It now takes a single argument 'args', which is a tuple containing all necessary arguments.
    """
    try:
        ticker, cg_map, cg_api, offset, window, lookahead = args
        print(f"Processing {ticker}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Running the asynchronous function to completion
        processed_asset = loop.run_until_complete(process_single_file(cg_map, cg_api, ticker))

        label_data = process_label_and_images_df(ticker, processed_asset, offset, window, lookahead)
        return label_data
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return pd.DataFrame()  
    
    

def create_all_labels_and_images_parallel(asset_list, cg_map, cg_api, offset, window, lookahead, num_processes=4):
    """
    Processes all assets in parallel using multiprocessing with a tqdm progress bar.
    """
    # Prepare arguments for each process
    args = [(ticker, cg_map, cg_api, offset, window, lookahead) for ticker in asset_list]
    # Use process_map instead of Pool.map to get the tqdm progress bar
    datasets = process_map(process_asset_parallel, args, max_workers=num_processes, chunksize=1)
    # Concatenate all DataFrames into a single DataFrame
    datasets_df = pd.concat(datasets, ignore_index=True)

    return datasets_df


async def create_all_labels_and_images(asset_list, cg_map, cg_api, offset, window, lookahead):
    """
    Processes all assets sequentially with a tqdm progress bar.
    """
    datasets = []
    for i in trange(len(asset_list), desc="Processing assets"):
        ticker = asset_list[i]
        processed_asset = await process_single_file(cg_map, cg_api, ticker)
        dataset = process_label_and_images_df(ticker, processed_asset, offset, window, lookahead)
        datasets.append(dataset)

    # Concatenate all DataFrames into a single DataFrame
    datasets_df = pd.concat(datasets, ignore_index=True)
    return datasets_df

# Example usage
if __name__ == "__main__":
    # Initialization code, ensure cg_map, cg_api, and other variables are defined
    asset_list = extract_asset_names('./raw_data')
    cg_map = map_cg_assets(asset_list, './crypto/coin_map.json')
    api_key = os.getenv('COINGECKO_API_KEY')
    cg_api = CoinGeckoAPI(api_key=api_key)
    offset = 240
    window = 16
    lookahead = 8
    # Call the function with multiprocessing
    # output_df = asyncio.run( create_all_labels_and_images(asset_list, cg_map, cg_api, offset, window, lookahead))
    output_df = create_all_labels_and_images_parallel(asset_list, cg_map, cg_api, offset, window, lookahead)
    output_df = output_df.sort_values('Start_Date')
    # Define file paths
    output_file_path = f'./crypto/data/{offset}_{window}_{lookahead}_labels_{datetime.now().strftime("%Y%m%d%H%M%S")}.feather'
    images_dat_path = f'./crypto/data/{offset}_{window}_{lookahead}_images_{datetime.now().strftime("%Y%m%d%H%M%S")}.dat'
    # Save DataFrame without images
    output_df.drop('Image', axis=1).to_feather(output_file_path)

    # Extract and save images
    images = output_df['Image'].tolist()
    with open(images_dat_path, 'wb') as file:
        pickle.dump(images, file)

    print(f"Data saved to {output_file_path} and {images_dat_path}")
    
# loaded_df = pd.read_feather('./crypto/data/240_16_8_labels.feather')
# images = pickle.load(open('./crypto/data/240_16_8_images.dat', 'rb'))