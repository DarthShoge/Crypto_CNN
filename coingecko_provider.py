import asyncio
from copy import copy
from datetime import datetime
from math import ceil
from ssl import SSLError
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import os
import requests
import json
import time
from functools import reduce
from enum import Enum

import aiohttp
from aiohttp import ClientSession


to_ts = lambda tss: [pd.Timestamp(x / 1000, unit="s") for x in tss]


class Exclude(Enum):
    STABLES = "stablecoins"
    WRAPPED_TOKENS = "wrapped-tokens"
    ETH_STAKING = "eth-2-0-staking"
    LIQUID_STAKING = 'liquid-staking-tokens'
    WORMHOLE_ASSETS = 'wormhole-assets'
    ASSET_BACKED_TOKENS = 'asset-backed-tokens'

    
def extract_raw_data(data, col_name):
    ts = [pd.Timestamp(x[0] / 1000, unit="s") for x in data]
    p = [x[1] for x in data]
    value_df = pd.DataFrame(index=ts, data=p, columns=[col_name])
    return value_df



class CoinGeckoAPI:
    
    def __init__(self, url = 'https://pro-api.coingecko.com/api/v3' , api_key:str = None) -> None:
        self.url = url
        self.api_key = {"x-cg-pro-api-key": api_key}
        self.session = None 
        # self.session = aiohttp.ClientSession(headers=self.api_key)
        
    async def fetch(self, url: str) -> dict:
        async with self.session.get(url, headers=self.api_key) as response:
            return await response.json()

    async def fetch_all(self, urls: List[str]) -> List[dict]:
        return await asyncio.gather(*(self.fetch(url) for url in urls))
    
        
    def get_exclusion_list(self, e: Exclude) -> List[str]:
        req = requests.get(
            f"{self.url}/coins/markets?vs_currency=usd&category={e.value}"
        , headers=self.api_key)
        data = req.json()
        return [x["id"] for x in data]
    
    def get_all_exlusions(self, excludes: List[Exclude] = []) -> List[str]:
        excludes = excludes if any(excludes) else list(Exclude)
        all_excludes = [self.get_exclusion_list(x) for x in excludes ]
        return reduce(lambda x, y: x+y, all_excludes )

    def raw_data_to_dataframe(self, inst_name : str, instr: object) -> pd.DataFrame:
        mcaps = extract_raw_data(instr["market_caps"], f"{inst_name}*mcap*")
        prices = extract_raw_data(instr["prices"], f"{inst_name}*price*")
        volumes = extract_raw_data(instr["total_volumes"], f"{inst_name}*volume*")
        return reduce(lambda left, right: left.join(right), [prices, mcaps, volumes])

    async def get_multi_asset_market_data(self, ids, data_days='max'):
        urls = [f"{self.url}/coins/{i}/market_chart?vs_currency=usd&days={data_days}&interval=daily" for i in ids]
        async with ClientSession() as self.session:
            responses = await self.fetch_all(urls)
        return {id_: resp for id_, resp in zip(ids, responses)}
    
    def get_ohlc_data(self, id, data_days='max'):
        url = f"{self.url}/coins/{id}/ohlc?vs_currency=usd&days={data_days}&interval=daily"
        data = requests.get(url, headers=self.api_key).json()
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        df['time'] = to_ts(df['time'])
        df.set_index('time', inplace=True)
        mkt_data_url = f"{self.url}/coins/{id}/market_chart?vs_currency=usd&days={data_days}&interval=daily"
        mkt_data = requests.get(mkt_data_url, headers=self.api_key).json()
        mkt_data_df = pd.DataFrame(mkt_data['total_volumes'], columns=['time', 'total_volumes'])
        mkt_data_df['time'] = to_ts(mkt_data_df['time'])
        mkt_data_df.set_index('time', inplace=True)
        df['volume'] = mkt_data_df.join(df, how='inner')['total_volumes']
        return df

    def get_market(self, limit: int = 200) -> List[str]:
        pages = int(ceil(limit / 200))
        current_market = [
            requests.get(
                    f"{self.url}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=200&page={x}&sparkline=false"
                , headers=self.api_key).json()
            for x in range(1, pages + 1)
        ]
        market_ids = [x["id"] for x in reduce(lambda l, r: l + r, current_market)]
        return market_ids[0:limit]
    
    def get_usd_price(self, ticker):
        asset_map = copy(self.get_symbol_map())
        print(f'assets:{len(asset_map)}')
        ids = [k for k,v in asset_map.items() ]
        assets = list(set(ids) - set(self.get_all_exlusions()))
        print(f'assets diff:{len(assets)}')
        rev_id = {asset_map[k]: k for k in assets if k in asset_map}
        data = requests.get(f"https://api.coingecko.com/api/v3/simple/price", params={'ids': rev_id[ticker], 'vs_currencies': 'usd'}).json()
        return float(data[rev_id[ticker]]['usd'])
    
    def get_symbol_map(self) -> Dict[str, str]:
        results = requests.get( f"{self.url}/coins/list", headers=self.api_key).json()
        asset_map = {x['id'] : x['symbol'].upper() for x in results }
        return asset_map

    def get_exclusions(self, exclusions: List[Exclude]) -> List[str] :
        excluded_lists = [ self.get_exclusion_list(x) for x in exclusions]
        excluded_ids = reduce(lambda l,r: l+r, excluded_lists)
        return excluded_ids

    def get_latest_price_data(self, coin_ids : List[str]) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame] :
        format_df = lambda ser: ser.to_frame().transpose().reindex(coin_ids, axis=1)
        now_time = datetime.now()
        req = self.url +'/simple/price?ids={}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true'.format(','.join(coin_ids)) 
        result = requests.get(req, headers=self.api_key).json()
        price_ser = pd.Series({k:v['usd'] for k,v in result.items()})
        mcap_ser = pd.Series({k:v['usd_market_cap'] for k,v in result.items()})
        vol_ser = pd.Series({k:v['usd_24h_vol'] for k,v in result.items()})
        price_ser.name = now_time
        mcap_ser.name = now_time
        vol_ser.name = now_time
        return format_df(price_ser), format_df(mcap_ser), format_df(vol_ser)

default_exlusions = [
        Exclude.STABLES,
        Exclude.ETH_STAKING,
        Exclude.WRAPPED_TOKENS,
        Exclude.LIQUID_STAKING
    ]

async def get_datapoints(ids, data_days, api: CoinGeckoAPI):
    all_data = await api.get_multi_asset_market_data(ids, data_days)
    all_dfs = [api.raw_data_to_dataframe(k,v) for k,v in all_data.items()]
    df_merged = reduce(lambda left, right: left.join(right), all_dfs)
    return df_merged


def extract_labelled_data(merged_df, label):
    value_df = merged_df[[col for col in merged_df.columns if label in col]]
    return value_df.ffill().rename(
        dict([(col, col.replace(label, "")) for col in value_df.columns]), axis=1
    )


async def get_cross_sectional_data(
    market_size: int = 220,
    exclusions: List[Exclude] = [
        Exclude.STABLES,
        Exclude.ETH_STAKING,
        Exclude.WRAPPED_TOKENS,
        Exclude.ASSET_BACKED_TOKENS,
        Exclude.LIQUID_STAKING
    ],
    from_date: Optional[datetime] = None,
    api_key = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches full historical timeseries for price, market cap and volume using coingecko api

    Args:
        market_size (int, optional): size of market e.g. top 200. Defaults to 220.
        exclusions (List[Exclude], optional): list of exclusion enums to filter from market. Defaults to [ STABLES, ETH_STAKING, WRAPPED_TOKENS, LIQUID_STAKING ].
```````````````````````````
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: price, market caps, volumes
    """
    api_key = get_api_key('coinGeckoAPIKey') if api_key is None else api_key
    api = CoinGeckoAPI(api_key=api_key)
    data_days = (datetime.now() - from_date).days if from_date else 'max'
    
    excludes = api.get_exclusions(exclusions)
    market = api.get_market(market_size)
    s = set(excludes)
    # Get exclusions
    excluded_market = [
        x for x in market if x not in s
    ]  # Not the most efficient but we get to keep the order
    
    #R emove duplicates
    excluded_market = list(dict.fromkeys(excluded_market))
    
    print(f'{len(market) - len(excluded_market)} tokens filtered out')
    merged_df = await get_datapoints(excluded_market, data_days, api)
    # Get latest slice of data
    lst_price_df, lst_mcap_df, lst_vol_df = api.get_latest_price_data(excluded_market)
    
    # Extract data and replace last slice with latest data fetch
    price_df = extract_labelled_data(merged_df, "*price*")
    price_df = pd.concat([price_df.iloc[:-1], lst_price_df])
    volume_df = extract_labelled_data(merged_df, "*volume*")
    volume_df = pd.concat([volume_df.iloc[:-1], lst_vol_df])
    mcap_df = extract_labelled_data(merged_df, "*mcap*")
    mcap_df = pd.concat([mcap_df.iloc[:-1], lst_mcap_df])
    return price_df, mcap_df, volume_df


async def get_single_asset_data(id, cg_api: CoinGeckoAPI, data_days='max'):
    merged_df = await get_datapoints([id], data_days, cg_api)
    price_df = extract_labelled_data(merged_df, "*price*")
    volume_df = extract_labelled_data(merged_df, "*volume*")
    mcap_df = extract_labelled_data(merged_df, "*mcap*")
    mcap_df = mcap_df.rename({id: 'mcap'}, axis=1)
    return price_df, mcap_df, volume_df
    

def get_snapshot_data(size, excludes) :
    price, mcaps, volume_df = get_cross_sectional_data()
    return json.dumps({
        'MarketCaps': mcaps.to_json(),
        'Prices': price.to_json(),
        'Volumes': volume_df.to_json()
    })
    


# #================================DELETE MEEEE!!!!!

# from multi_asset import get_api_key
# pf_data = requests.get('https://func-dev-uks-positionsexposures.azurewebsites.net/api/ExposureReport?portfolio=Lukman_Test').json()
# positions = pd.read_json( pf_data['positions'])
# position_prices = positions[['symbol','asset','avgPx','markPx','venue']]
# cg_api = CoinGeckoAPI(api_key=get_api_key())
