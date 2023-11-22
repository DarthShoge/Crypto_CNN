import json
import os
import pandas as pd
import dotenv
import asyncio
import pickle
from datetime import datetime

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


def process_label_df(ticker, asset_df, offset, window, lookahead):
    full_period = window + lookahead

    result_df = pd.DataFrame(columns=[
    'Asset',
    'Start_Date', 
    'End_Date', 
    'Daily_Return', 
    f'Ret_{full_period}H', 
    f'Ret_{lookahead}H', 
    'Market_Cap', 
    'Image'])



    for i in range(offset, len(asset_df) - full_period, full_period):
    # Define the period for the current window
        start_date = asset_df.iloc[i]['Timestamp']
        end_date = asset_df.iloc[i + window - 1]['Timestamp']
        lookahead_end_date = asset_df.iloc[i + full_period - 1]['Timestamp']

    # Get the current block of data
        window_data = asset_df.iloc[i:i + window]

    # Generate the image
        image = generate_price_image(window_data, (64, 48))
    
    # Calculate returns
        start_price = asset_df.iloc[i]['Close']
        market_cap = asset_df.iloc[i]['mcap']
        end_price = asset_df.iloc[i + window - 1]['Close']
        lookahead_price = asset_df.iloc[i + full_period - 1]['Close']
    
        daily_return = (end_price - start_price) / start_price
        full_period_return = (lookahead_price - start_price) / start_price
        lookahead_return = asset_df.iloc[i + window: i + full_period]['Close'].pct_change().sum()

    # Store the results in the new DataFrame
        result_df = result_df._append({
        'Asset': ticker, 
        'Start_Date': start_date, 
        'End_Date': lookahead_end_date, 
        'Daily_Return': daily_return, 
        f'Ret_{full_period}H': full_period_return, 
        f'Ret_{lookahead}H': lookahead_return, 
        'Market_Cap': market_cap,	
        'Image': image
    }, ignore_index=True)
    return result_df

def process_asset(args):
    """
    Wrapper function for processing a single asset.
    This function will be called in parallel for different tickers.
    It now takes a single argument 'args', which is a tuple containing all necessary arguments.
    """
    try:
        ticker, cg_map, cg_api, offset, window, lookahead = args
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Running the asynchronous function to completion
        processed_asset = loop.run_until_complete(process_single_file(cg_map, cg_api, ticker))

        label_data = process_label_df(ticker, processed_asset, offset, window, lookahead)
        return label_data
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return pd.DataFrame()  
    
    

def create_all_labels_and_images(asset_list, cg_map, cg_api, offset, window, lookahead, num_processes=4):
    """
    Processes all assets in parallel using multiprocessing with a tqdm progress bar.
    """
    # Prepare arguments for each process
    args = [(ticker, cg_map, cg_api, offset, window, lookahead) for ticker in asset_list]

    # Use process_map instead of Pool.map to get the tqdm progress bar
    datasets = process_map(process_asset, args, max_workers=num_processes, chunksize=1)

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
    output_df = create_all_labels_and_images(asset_list, cg_map, cg_api, offset, window, lookahead, num_processes=8)
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
    
loaded_df = pd.read_feather('./crypto/data/240_16_8_labels.feather')
images = pickle.load(open('./crypto/data/240_16_8_images.dat', 'rb'))