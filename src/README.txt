Python Version: 3.5 or greater
filename: logreg.py

Steps:

Step-1:
Run the program as:
python3 logreg.py

step-2:
It will prompt for the csv dataset file name. e.g: <filename>.csv

step-3:
Updated weights are stored in weights.csv file.

step-4:
It plots the two graph.
1. epochs vs sum of square error
2. Decision boundary graph( Input attribute 1 samples vs Input attribute 1 samples )

step-5:
It also prints the number of correct and incorrect classified samples for each class


- Different dataset requires different value of epochs and learning rate.
- Please change those values at line number 187 and 188( Inside class LogisticRegression )
- Input data are stored inside the program as [ 1, x1, x2 ] and weight as [ bias, w1, w2 ]
