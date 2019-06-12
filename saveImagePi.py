import time
from time import gmtime, strftime
import urllib.request
starttime=time.time()
i = 0
while True:
  currentTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
  print("saving image @"+str(currentTime))
  urllib.request.urlretrieve("https://us-central1-core-228912.cloudfunctions.net/raspberry-pi-a7?apiKey=MFDW!!HQ_!CCPK?Q?QJEGAKQ!ENFM<>(XGQ}BXAMM<Q:", "img")
  time.sleep(20.0 - ((time.time() - starttime) % 20.0))
  print("Image saved!")
  i=i+1
