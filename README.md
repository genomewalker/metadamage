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

Then run as:
###### 3)
```console
$ MADpy --help
$ MADpy ./data/input/data_ancient.txt
$ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt
```

MADpy also allows for fitting and plotting multiple files (and automatically compares their fit results):
###### 4)
```console
$ MADpy --verbose --number_of_fits 10 --num_cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt
```



