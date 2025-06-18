# vision-core
## Hardware - minimum:
- Rasberry Pi 4 (4gb)
- SD Card 32gb
- CAMERA IP With RTSP
- Stable internet (wifi) connection
## Software:
- Raspberry Pi OS (Bullseye or newer)
- Python 3.10.11 is recommended
## Image Capturing system -> camera_cronjob.py
## AI Analyze syste, -> analyze_image.py
## Convulsions Tracking system -> video_streaming.py

##How to start
1. Use pyenv for mac/linux, uv for window to make sure we use correct version 3.10.11
2. use venv to isolate the python env (python venv .venv310)
3. source .venv310/bin/activate (Mac) .venv310/Script/activate (Win)
4. pip install -r requirements_full.txt
5. Check the env file (Especially the rtspurl and the server to run should be the same network with the wifi to run rtspurl for the convulsion detection )
6. u can  run all by using  python run_all_services.py 
7. Or run 1 by 1 like (mostly for debug, and run at the same time for 3 commands as well): 
   + mera_cronjob.py: python camera_cronjobs.py
   + analyze_image.py: python analyze_image.py
   + video_streaming.py: python video_streaming.py "rtsp://admin:pwd@ip:554/cam/realmonitor?channel=1&subtype=1" --image_id "1750176801"
Hope u like it. In this demontration because we only have more than 30hrs so:
1. I'm using imou's api for capturing (fast)
2. Using LLM models (OpenAI Assistant to make it fast - we gonna use Claude soon :P )
3. limitation about network. I have to use 4g for the camera network, that's why the connection a bit slow for supabase data sync up

To see a result please checkout this source code and build apk (or just using a emulator): https://github.com/trieuaiinsight/vision-mobile-app