# pip install SoccerNet --upgrade

# import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
sn = SoccerNetDownloader(LocalDirectory="path/to/SoccerNet")
# sn.password = "s0cc3rn3t"

# # Ball Action Spotting: videos y labels
# sn.downloadDataTask(task="spotting-ball-2023", split=["train","valid","test","challenge"])
# # o en algunos setups OSL:
# # sn.downloadDataTask(task="spotting-OSL", split=["train","valid","test","challenge"], version="224p")  # requiere password
# # Para vídeos explícitos por partidos/mitades:
# sn.downloadGames(files=["1_224p.mkv","2_224p.mkv"], split=["train","valid","test"])
# # (cambiá a 720p si querés alta resolución)

# # Etiquetas + vídeos de Ball Action Spotting (ajustá el nombre de task si tu versión usa otro alias)
# sn.downloadDataTask(
#     task="spotting-ball-2023",               # o "spotting-OSL" según versión
#     split=["train", "valid", "test"],
#     password="s0cc3rn3t"                     # la del mail del NDA
# )

sn.password = "s0cc3rn3t"  # del mail del NDA
sn.downloadGames(
    files=["1_224p.mkv","2_224p.mkv"],  # o ["1_720p.mkv","2_720p.mkv"]
    split=["train","valid","test"]
)