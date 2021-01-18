## Metadamage - Metagenomics Ancient Damage


If first time using:

###### 1a)
```console
$ git clone https://github.com/ChristianMichelsen/metadamage
$ cd metadamage
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
$ conda activate metadamage
```

Then run as:
###### 3)
```console
$ metadamage --help
$ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt
```

metadamage also allows for fitting and plotting multiple files (and automatically compares their fit results):
###### 4)
```console
$ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt ./data/input/data_control.txt
```



