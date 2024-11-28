# Strong-Field Exercises, One-Dimensional TDSE

## Running instructions

Make sure you have Python 3 installed. Then run the following to
create a virtual environment and install the necessary dependencies:
```
$ python3 -m venv venv
$ source venv/bin/activate
venv $ python -m pip install -r requirements.txt
```

Then start `ipython` and enable the `matplotlib` magic:
```
venv $ ipython
In[1]: %matplotlib
```
We can then run one of the provided calculations as
```
In[2]: %run single.py inputs_part1/case1.1.ini
```
