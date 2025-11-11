# pip install SoccerNet --upgrade

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="path/to/SoccerNet")
downloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])