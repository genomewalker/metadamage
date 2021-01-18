## Metagenomics Ancient Damage python: MADpy-pkg


If first time using:

###### 1a)
```console
$ git clone https://github.com/ChristianMichelsen/MADpy-pkg
$ cd MADpy-pkg
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
$ conda activate MADpy-pkg
```

Then run as:
###### 3)
```console
$ MADpy-pkg --help
$ MADpy-pkg ./data/input/data_ancient.txt
$ MADpy-pkg --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt
```

MADpy-pkg also allows for fitting and plotting multiple files (and automatically compares their fit results):
###### 4)
```console
$ MADpy-pkg --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt
```



