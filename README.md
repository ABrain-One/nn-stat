# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> Neural Network Performance Analysis
<sub><a href='https://pypi.python.org/pypi/nn-stat'><img src='https://img.shields.io/pypi/v/nn-stat.svg'/></a><br/>short alias <a href='https://pypi.python.org/pypi/lmurs'>lmurs</a></sub>

The original version of the NN Stat project was created by <strong>Waleed Khalid</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

<img src='https://abrain.one/img/lemur-nn-stat-whit.jpg' width='25%'/>

<h3>Overview 📖</h3>

<p>Automated conversion of <a href="https://github.com/ABrain-One/nn-dataset" target="_blank" rel="noopener noreferrer">LEMUR</a> data into Excel format with statistical visualizations. It is developed to support the <a href="https://github.com/ABrain-One/nn-dataset">NN Dataset</a> and <a href="https://github.com/ABrain-One/nn-gpt">NNGPT</a> projects.</p>

## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   ```

It is assumed that CUDA 12.6 is installed. If you have a different version, please replace 'cu126' with the appropriate version number.

## Environment for NN Stat Contributors

Run the following command to install all the project dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Installation with the LEMUR Dataset

```bash
pip install nn-stat[dataset]
```

## Usage

```bash
python -m ab.stat.export
```
Data and statistics are stored in the <strong>stat</strong> directory in Excel files and PNG/SVG plots.

To use 'ab/stat/nn_analytics.ipynb' install jupyter:

```bash
pip install jupyter
```

and run jupyter notebook:

```bash
jupyter notebook --notebook-dir=.
```

## Update of NN Dataset
Remove old version of the LEMUR Dataset and its database:
```bash
pip uninstall nn-dataset -y
rm -rf db
```
Install from GitHub to get the most recent code and statistics updates:
```bash
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```
Installing the stable version:
```bash
pip install nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```


### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:. abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python -m ab.stat.export"
```
Some recently added dependencies might be missing in the <b>AI Linux</b>. In this case, you can create a container from the Docker image ```abrainone/ai-linux```, install the missing packages (preferably using ```pip install <package name>```), and then create a new image from the container using ```docker commit <container name> <new image name>```. You can use this new image locally or push it to the registry for deployment on the computer cluster.

#### The idea and leadership of Dr. Ignatov
