# HW3 Setup Instructions

### Konrad Kraemer (kkraemer6)

## Data Sources

1) Wine: https://archive.ics.uci.edu/ml/datasets/wine+quality

__Note__: Only download the `winequality-white.csv` file, that's what this analysis uses

2) Car: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

## Running this script

__Note__: These scripts were written using python3 and pip3.

1) Ensure python3 is installed on your computer
2) Install the requirements using the steps below:
`pip3 install numpy`
`pip3 install pandas`
`pip3 install sklearn`
`pip3 install scipy`
`pip3 install IPython`
`pip3 install matplotlib`
`pip3 install pydotplus`
`pip3 install graphviz`
11) Once all requirements are installed, run `python3 main.py`

## Preprocessing the Wine dataset
1) Download the dataset and open it in a text-editor of your choice
2) Using find-and-replace, change all the ";" to "," to fix the formatting and make it a true CSV file
3) Save this file as `winequality-data.csv` 

## Preprocessing the Car Dataset
1) Download the dataset and open it in excel
2) Add a row at the top for the attribute names
3) Fill in this top row: `buying, maint, doors, persons	lug_boot, safety, acceptability`
4) Save this file as `car-data.csv`

## Code:
The code can be found at the following address. Included are also all of my plots:
https://github.com/kraemerk/CS4641hw3