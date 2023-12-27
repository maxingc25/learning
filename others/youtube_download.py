
import yt_dlp

#Enter the url the download 
url = input("Enter video url: ")

ydl_opts = {}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print ("Video downloaded successfully!")
