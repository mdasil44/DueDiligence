import os
import pandas as pd
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = 'Tpk_c8897b1c65f241558f5d4d5477241552'

yrs = 5
startDate = datetime.now() - timedelta(days=yrs*365)
currDate = datetime.now()

stock = get_historical_data("XOM", start=startDate, end=currDate, output_format='pandas')
