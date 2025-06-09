# Evaluation

Make sure you have the scenes downloaded and placed in `assets/models/Scenes/`!

## Setup

Evaluation was run on an AMD Ryzen 9 7950X 16-Core Processor with an NVIDIA GeForce RTX 4090 (24 GB of VRAM) running Ubuntu. __It is recommended you run this on a machine with ample resources (many cores, high VRAM).__

Tested with Python 3.10 and up.

### Install dependencies

```
pip3 install requirements.txt
```

## Run Evaluation

To run the evaluation described in the paper, you can run:
```
python3 run_eval.py 20 10 --pose-prediction                   # run 20+/-10ms trace (w/ pose prediction)
python3 run_eval.py 50 20 --pose-prediction --pose-smoothing  # run 50+/-20ms trace (w/ pose prediction and smoothing)
```
These will run traces (found in `../assets/paths/`) for the Robot Lab, Sun Temple, Viking Village, and San Miguel scenes for 0.25m, 0.5m, and 1.0m viewcell sizes.

__WARNING__: these scripts will take a while to run and will use a lot of resources on your computer!

### Optional Paramters

* If you want to run shorter traces (500 frames instead of 1500), you can add `--short-paths`.

* If you want to run with specific viewcell size(s), you can add `--view-sizes <comma-seperated list>`.

* If you want to run with specific scene(s), you can add `--scenes <comma-seperated list>`.

Example (this will run the Robot Lab scene with a short trace with viewcell sizes of 0.5m and 1.0m):
```
python3 run_eval.py 20 10 --pose-prediction --short-paths --view-sizes 0.5,1.0 --scenes robot_lab
```

See `run_eval.py` for more command line parameters.

### Results

Results will be packed in tarball files in the `results/` folder:
```
results/
├── results_20.0_10.0ms_0.25m.tar.gz  # results with 20+/-10ms of latency with a 0.25m viewcell size
├── results_20.0_10.0ms_0.5m.tar.gz   # results with 20+/-10ms of latency with a 0.5m viewcell size
...                                   # etc.
```

Untarring the files will reveal:
```
results_20.0_10.0ms_0.25m/
├── errors.json                       # json file containing FLIP, SSIM, and PSNR errors for each method
└── results/
    ├── stats/
    │   └── robot_lab/
    │       ├── qr_simulator.log
    │       ├── qs_simulator.log
    │       ├── ...
    │       ├── scene_viewer.log
    │       └── stats.json            # json file containing performance timings and data payload statistics
    └── videos/
        └── robot_lab/
            ├── color/                # color videos for ground truth (scene_viewer) and all tested methods
            └── flip/                 # FLIP error map videos for all tested methods
```
