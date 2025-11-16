"""Download the SoccerNet Ball Action Spotting videos and labels.

The SoccerNet download password is under an NDA and must not be committed.
Export it before running this script:

    export SOCCERNET_PASSWORD='...'
"""

import os

from SoccerNet.Downloader import SoccerNetDownloader

LOCAL_DIRECTORY = os.environ.get("SOCCERNET_DIR", "path/to/SoccerNet")
PASSWORD = os.environ.get("SOCCERNET_PASSWORD")

if not PASSWORD:
    raise SystemExit("SOCCERNET_PASSWORD is not set. Export it and run again.")

sn = SoccerNetDownloader(LocalDirectory=LOCAL_DIRECTORY)
sn.password = PASSWORD

# Explicit per-game halves. Swap 224p for 720p to get the high-resolution videos.
sn.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"],
    split=["train", "valid", "test"],
)

# Alternative: pull the Ball Action Spotting labels and videos as a single task.
# Depending on the SoccerNet version the task alias may be "spotting-OSL".
#
# sn.downloadDataTask(
#     task="spotting-ball-2023",
#     split=["train", "valid", "test", "challenge"],
# )
