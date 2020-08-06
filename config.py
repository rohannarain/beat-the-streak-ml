import os

import datetime
import pytz

today = datetime.datetime.today()
local_tz = pytz.timezone("America/Los_Angeles")
today = today.replace(tzinfo=pytz.utc).astimezone(local_tz)


class Config:
	SECRET_KEY = os.getenv("SECRET_KEY") or "something-else"
	TODAY_PACIFIC_TIME = today.strftime("%m_%d_%Y")
