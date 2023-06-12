import pytz
from datetime import datetime

def korea_date_time():
    """
    Retrieves the current date and time in the Korea Standard Time (KST) timezone.

    Returns:
        str: The current date and time formatted as 'YYYY-MM-DD_HH:MM:SS' in KST.
    """
    korea_timezone = pytz.timezone("Asia/Seoul")
    date_time = datetime.now(tz=korea_timezone)
    date_time = date_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    return date_time
