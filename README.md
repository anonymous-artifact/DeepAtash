# DeepAtash

Focused test generation for DL systems

## General Information ##
This folder contains the application of the DeepAtash  approach to the handwritten digit classification problem and movie sentiment analysis.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.8.

## Dependencies ##


### Installing Python 3.8 ###
Install Python 3.8:

``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.8
```

And check if it is correctly installed, by typing the following command:

``` 
python3 -V

Python 3.8.5
```

Check that the version of python matches `3.6.*`.

### Installing pip ###

Use the following commands to install pip and upgrade it to the latest version:

``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version

pip 21.1.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
```
### Creating a Python virtual environment ###

Install the `venv` module in the docker container:

``` 
apt install -y python3-venv
```

Create the python virtual environment:

```
cd /DeepHyperion-MNIST
python3 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```

### Installing Python Binding to the Potrace library ###
Install Python Binding to the Potrace library.

``` 
apt install -y build-essential python-dev libagg-dev libpotrace-dev pkg-config
``` 

Install `pypotrace` (commit `76c76be2458eb2b56fcbd3bec79b1b4077e35d9e`):

``` 
cd /
git clone https://github.com/flupke/pypotrace.git
cd pypotrace
git checkout 76c76be2458eb2b56fcbd3bec79b1b4077e35d9e
pip install numpy
pip install .
``` 

To install PyCairo and PyGObject, we follow the instructions provided by [https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started](https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started).

``` 
apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
apt install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev
``` 

## MNIST or IMDB ## 

### Installing Other Dependencies ###

This tool has other dependencies, including `tensorflow` and `deap`, that can be installed via `pip`:

```
cd /DeepAtash/MNIST
```
or
```
cd /DeepAtash/IMDB
```
``
pip install -r requirements.txt
``` 

## Usage ##
### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* A list of seeds used for the input generation. In this implementation, the seeds are indexes of elements of the MNIST dataset. The default list is in the file `bootstraps_five`;
* `config.py` containing the configuration of the tool selected by the user.

### Run the Tool ###

```
python ga_method.py [RUN_ID]
```
or 
```
python nsga2_method.py [RUN_ID]
``` 

