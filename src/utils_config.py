import datetime
import pytz

today = datetime.datetime.today()
local_tz = pytz.timezone("America/Los_Angeles")
today = today.replace(tzinfo=pytz.utc).astimezone(local_tz)
yesterday = today - datetime.timedelta(days=1)

class Config:
	_conf = {
		"curr_season": 2021,
		"curr_season_start": "04/01/2021",
		"curr_season_end": "10/03/2021",
		"YESTERDAY": yesterday.strftime("%m/%d/%Y"),
		"TODAY": today.strftime("%m/%d/%Y"),
	}
	_setters = ["curr_season", "curr_season_start", "curr_season_end"]

	@staticmethod
	def get(name):
		return Config._conf[name]

	@staticmethod
	def set(name, value):
		if name in Config._setters:
			Config._conf[name] = value
		else:
			raise NameError("Name not accepted in set() method")

