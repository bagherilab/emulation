rm -rf results/ARCADE/*
python src/config.py cs=coupled-c
python src/config.py cs=coupled-chx
python src/config.py cs=uncoupled-c
python src/config.py cs=uncoupled-chx
python src/config.py cs=static-c
python src/config.py cs=static-chx
