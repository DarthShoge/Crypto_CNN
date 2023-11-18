import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from pathlib import Path
from tqdm import tqdm

def create_ohlc_image(prices, config):
    """
    Create an OHLC image based on the provided prices and configuration.

    :param prices: DataFrame with columns for open, high, low, close, vol, and ma20
    :param config: Dictionary for configuration parameters
    :return: Numpy array containing pixel values for OHLC image
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                   gridspec_kw={'height_ratios': [1, config['volume_height_ratio']]},
                                   figsize=(len(prices) * config['width_factor'], config['height']),
                                   facecolor='black', dpi=config['dpi'])

    ax1.plot(prices.index, prices['ma'], color='white', linewidth=0.5)
    ax1.vlines(prices.index, prices['Low'], prices['High'], color='white', linewidth=1)
    ax1.scatter(prices.index, prices['Open'], marker='_', color='white', s=10)
    ax1.scatter(prices.index, prices['Close'], marker='_', color='white', s=10)

    ax2.bar(prices.index, prices['Volume'], color='white', width=0.5)

    for ax in [ax1, ax2]:
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout(pad=0)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)

    plt.close(fig)

    return image


# Configuration with reduced resolution and size
config = {
    'height': 3,          # Adjusted height for the image
    'width_factor': 0.25, # Adjusted width factor to shrink the image further
    'volume_height_ratio': 0.2,
    'file_format': 'png',
    'dpi': 50            # Adjusted DPI for reduced resolution
}

def generate_ohlc_images_and_labels(ohlcs_data, asset_name, lookback, config):
    """
    Generate OHLC images and labels for a given asset and lookback period.
    
    :param ohlcs_data: DataFrame containing OHLC data with a 'Date' column
    :param asset_name: String representing the asset's name, e.g., 'AAPL'
    :param lookback: Integer representing the lookback window size, e.g., 20
    :return: DataFrame containing file paths and corresponding labels
    """
    
    # Create a directory for the asset with lookback
    asset_dir = Path(f'{asset_name}_{lookback}')
    asset_dir.mkdir(exist_ok=True)
    
    labels_data = []

    # Generate images and labels
    for i in tqdm(range(lookback, len(ohlcs_data))):
        data = ohlcs_data.iloc[i - lookback:i]
        
        # Generating the filename with the date format as specified
        from datetime import datetime

        start_date = datetime.strptime(data.iloc[0].name, '%d/%m/%Y').strftime('%Y%m%d')
        end_date = datetime.strptime(data.iloc[-1].name, '%d/%m/%Y').strftime('%Y%m%d')
        filename = f'{asset_name}_{start_date}_{end_date}.{config["file_format"]}'
        
        # Create the image path
        img_path = asset_dir / filename

        # Generate image
        # img = create_ohlc_image(data, config)
        # plt.imsave(str(img_path), img)
        
        # # Calculate the return and the label
        # future_close = ohlcs_data.loc[i, 'Close']  # Using current day close for labeling
        # ret = (future_close - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
        # label = 1 if ret > 0 else 0
        
        # Append to labels with additional data
        labels_data.append({
            'date': data.iloc[-1].name,
            'open': data['Open'].iloc[0],
            'high': data['High'].iloc[0],
            'low': data['Low'].iloc[0],
            'close': data['Close'].iloc[0],
            'volume': data['Volume'].iloc[0],
            'image_path': str(img_path),
            # 'label': label
        })

    # Convert labels data to DataFrame
    labels_df = pd.DataFrame(labels_data)
    
    # Store labels DataFrame as CSV in the asset directory
    csv_path = asset_dir / 'labels.csv'
    labels_df.to_csv(csv_path, index=False)

    return labels_df


def draw_line(image, start_y, start_x, end_y, end_x):
    ''' Draw a line in the image from start point to end point '''
    # Swap start and end points if start_x is greater than end_x
    if start_x > end_x:
        start_x, end_x = end_x, start_x
        start_y, end_y = end_y, start_y

    dx = end_x - start_x
    dy = end_y - start_y
    error = dx / 2.0
    y = start_y
    ystep = 1 if start_y < end_y else -1

    if dx > dy:
        for x in range(start_x, end_x + 1):
            image[y, x] = 255
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    else:
        for x in range(start_x, end_x + 1):
            image[y, x] = 255
            y += ystep

def generate_price_image(df, image_size, show_volume=True):
    """
    Generate a visual representation of stock price data and volume as a grayscale image
    to the specification of the (Re-)Imag(in)ing Price Trends paper.

    The function normalizes the OHLC (Open, High, Low, Close) and moving average (ma) values to fit
    within the image dimensions. If volume data is included, it is plotted at the bottom of the image.
    The price and moving average data are plotted above the volume section. The image is then inverted
    vertically so that the volume bars start from the bottom.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the columns 'Open', 'High', 'Low', 'Close', 'ma', and 'Volume'.
                         Each row corresponds to a time interval (e.g., a day).
    - image_size (tuple): A tuple (height, width) specifying the dimensions of the output image.
                          The height must be divisible by 4 if volume is to be shown, as 1/4 of the image's
                          height is reserved for volume bars. The width should be three times the number of
                          time intervals to be plotted, as each interval occupies three pixels.
    - show_volume (bool): If True, volume data is included in the image. Defaults to True.

    Returns:
    - image (np.array): A 2D NumPy array representing the image, with the price and moving average data occupying
                        the upper section and volume data (if included) occupying the bottom section. Each 'pixel'
                        in the image is a value of 0 (black) or 255 (white).

    Usage:
    - For a 5 period image, use a image size (32, 15):
    - For a 20 period image, use a image size (64, 60):
    - For a 60 period image, use a image size (96, 180):

    To generate an image for a subset of the data with volume:
    >>> subset_df = df.iloc[start_interval:end_interval]
    >>> img = generate_price_image(subset_df, (64, 60), show_volume=True)

    To generate an image for the same subset without volume:
    >>> img_no_vol = generate_price_image(subset_df, (64, 60), show_volume=False)

    Note:
    - The input DataFrame must have its indices reset if it is a subset of a larger DataFrame.

    """
    # Normalize OHLC values and ma20 to fit the image dimensions
    min_price = df[['Open', 'High', 'Low', 'Close', 'ma']].min().min()
    max_price = df[['Open', 'High', 'Low', 'Close', 'ma']].max().max()
    df_norm = (df[['Open', 'High', 'Low', 'Close', 'ma']] - min_price) / (max_price - min_price)

    # Normalize Volume to fit the designated area of the image
    volume_section_height = image_size[0] // 4
    price_section_height = image_size[0] - volume_section_height

    max_volume = df['Volume'].max()
    df_norm['Volume'] = (df['Volume'] / max_volume) * volume_section_height
    df_norm['Volume'] = df_norm['Volume'].apply(np.floor).astype(int)

    # Scale price data to fit the price section of the image
    df_norm[['Open', 'High', 'Low', 'Close', 'ma']] *= (price_section_height - 1)
    df_norm[['Open', 'High', 'Low', 'Close', 'ma']] = df_norm[['Open', 'High', 'Low', 'Close', 'ma']].apply(np.floor).astype(int)

    # Move the price plot up by adding an offset
    price_offset = volume_section_height
    df_norm[['Open', 'High', 'Low', 'Close', 'ma']] += price_offset

    # Prepare the image
    image = np.zeros(image_size)

    if show_volume:
        volume_base = image_size[0] - 1  # The bottom row of the image
        for i in range(len(df)):
            vol_p = int(df_norm.iloc[i]['Volume'])
            # Draw the volume from the bottom up
            image[volume_base - vol_p:volume_base, i * 3 + 1] = 255
            
    image = np.flipud(image)
    # Draw the candlesticks and MA
    for i in range(len(df)):
        open_p = df_norm.iloc[i]['Open']
        close_p = df_norm.iloc[i]['Close']
        high_p = df_norm.iloc[i]['High']
        low_p = df_norm.iloc[i]['Low']
        ma_p = df_norm.iloc[i]['ma']

        # Draw the high-low line
        image[low_p:high_p + 1, i * 3 + 1] = 255

        # Draw open and close prices as horizontal ticks
        image[open_p, i * 3] = 255
        image[close_p, i * 3 + 2] = 255

        # Draw the moving average line
        if i > 0:
            prev_ma_p = df_norm.iloc[i - 1]['ma']
            draw_line(image, prev_ma_p, (i - 1) * 3 + 1, ma_p, i * 3 + 1)

    return np.flipud(image)



