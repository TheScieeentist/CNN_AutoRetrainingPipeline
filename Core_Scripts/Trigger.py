from apscheduler.schedulers.background import BackgroundScheduler
import time
from Pipeline.main import main
from pytz import timezone

import logging


# This is the trigger file, which runs in the background and runs the main() function/script at a pre-defined date.

# setting up the scheduler using timezone to set the timezone to german time stadard:
scheduler = BackgroundScheduler(timezone=timezone('Europe/Berlin'))
scheduler.start()

# adjusting the scheduler to run the main() script at every tuesday moning at ____:
scheduler.add_job(main, 'cron', day_of_week='tue', hour=10, minute=9)

logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.DEBUG)

# this part defines the interruptions of the trigger.py, so that it runs in the background 
# as long as no interruption is called with the keyboard in the terminal:
try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    