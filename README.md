# Climate Storyline - storypy

## Using the package without Esmvaltool
#### Virtual environment with pip
```pip3 install virtualenv```

```python3 -m venv sp_venv```

```source sp_venv/bin/activate```

The virtual environment can deactivated by simply:

```deactivate```
_____________________________________________________________________

#### Virtual environment with conda
```conda create -n sp_env python=3.6.3```

```conda activate sp_env```

To deactivate, simply:

```conda deactivate```



## If installing to be used with Esmvaltool recipe (... skip this step for now)
Before installing the storypy python package, users need to setup Esmvaltool their machine. After the setup of Esmvaltool, activate the esmvaltool environment. Then, pip install the storypy package. This is because Esmvaltool uses some dependencies which are only available in conda.


Install the storypy package

```pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple storypy```


