Run the following to create the conda env

```
conda env create -f environment.yml
```

Then run the following for the pip dependencies

```
pip install -r requirements.txt
```

Then add locobotSim as a dev package with

```
pip install -e .
```

To run anything, first activate the env with

```
conda activate DesSocCog
```
