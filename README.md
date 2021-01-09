## Metagenomics Ancient Damage python: MADpy


If first time using:

###### 1a)
```console
$ git clone https://github.com/ChristianMichelsen/MADpy
$ cd MADpy
$ conda env create -f environment.yaml
```

or, if already installed:
###### 1b)
```console
$ conda env update --file environment.yaml
```

Afterwards remember to activcate the new environment:
###### 2)
```console
$ conda activate MADpy
```


Install as local package
###### 3)
```console
$ pip install --editable .
```

Then run as:
###### 4)
```console
$ MADpy --help
$ MADpy ./data/input/data_ancient.txt
```

