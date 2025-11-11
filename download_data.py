# pip install SoccerNet --upgrade

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
downloader =SoccerNetDownloader(LocalDirectory="path/to/SoccerNet")

downloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])

# optional: download ResNet features or Baidu features
downloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
                         split=["train","valid","test","challenge"])