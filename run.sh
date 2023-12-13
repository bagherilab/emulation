rm -rf results/ARCADE/*
python src/config.py cs=topological_0_metric_8
python src/config.py cs=topological_0_metric_15

python src/config.py cs=hemodynamic_0_metric_8
python src/config.py cs=hemodynamic_0_metric_15

python src/config.py cs=spatial_0_metric_8
python src/config.py cs=spatial_0_metric_15
