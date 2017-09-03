# A Single Logistic Regressor

### Problem Statement:

Implement and test a single logistic regressor.

### Description:

As we know that a logistic regressor is defined by a linear model whose (single) output value is fed through a logistic function (or sigmoid), producing a value in the interval (0; 1) as output. Logistic regressors are used for binary classification by rounding this output value (>= 0.5 is Class 1; otherwise Class 0).

### Steps:

1. Given a .csv file containing a list of samples in a 2D space (one per line; variable values before the target class value of 0 or 1 in the final column), the program fit the weight vector **'w'** for the linear model using batch training (i.e. iterating over the samples, and then updating weights after classifying each sample), and then save the weight values to a file **weights.csv**. The program then use matplotlib to produce a line graph showing epoch (pass over the entire training set) vs. the sum of squared error for all samples in each epoch.

2. Given a .csv file of samples and a weight vector file, the program produces a plot (using matplotlib) showing the samples with their true class from the file, and plots the decision boundary of the classifier as a line. Used color to make clear which samples are from Class 1, and which are from Class 0. The program then report the number of correctly and incorrectly classified samples for each class.

   ​

------

### Execution of program:

**Main file name:** logreg.py

**Python version:** 3.5.1

Run the python code as follows

```powershell
python logreg.py <filename.csv>
```

where,

| Parameter        | Details              |
| ---------------- | -------------------- |
| \<filename.csv\> | sample data csv file |

### Output:

1. **For Linearly Separable data:**

   ```powershell
   Number of correct class 0 sample:        25
   Number of incorrect class 0 sample:      0
   Number of correct class 1 sample:        25
   Number of incorrect class 0 sample:      0
   ```

   ![Linearly Separable data](https://github.com/karan6181/LogisticRegressor/blob/master/Outputs/Images/LinearlySeparable.png)

   From the above diagram, it is clearly seen that the data is linearly separable. There are no incorrectly classified samples for each class. Initially, sum of square error is quite high and it decreases as the number of epochs increases or as the machine learns from the data.

2. **For Non-linearly Separable data:**

   ```powershell
   Number of correct class 0 sample:        51
   Number of incorrect class 0 sample:      19
   Number of correct class 1 sample:        52
   Number of incorrect class 0 sample:      18
   ```

   ![Non-linearly Separable data](https://github.com/karan6181/LogisticRegressor/blob/master/Outputs/Images/nonLinearlySeparable.png)

   The above diagram shows that the data is non-linearly separable. That means there are some incorrectly classified samples for each class. The above model doesn’t completely represent by the linear decision boundary.
