using System;
using Accord.Math;


namespace MLFunctions
{ 
   class math
    { 
         static public double [,] Sigmoid(double [,] input)
        {
            //g = 1 ./ (1 + exp (-z));
            for (int i = 0; i < input.Rows(); i++)
                for (int j = 0; j < input.Columns(); j++)
                {
                    input[i, j] = 1/ (1 + Math.Exp(input[i, j]));
                }
            return input;
        } 

        /*static public double [,] Sigmoid (double [,] z)
        {
            // g = 1 / (1 + exp (-z))
            1 / (1 + Math.Exp(-z));

            return Accord.Math.Sigmoid(input);
            
        } */

        static public double L2Regularization(double [,] Theta, double lambda, long m)
        {
            // Regularization term excluding theta (0) Note both L1 and L2
            // regularization routines  are provided, comment/uncomment
            // as appropriate
            //           
            // L2 regularization uses the squared magnitude of coefficient
            // as penalty term to the loss function.
            // reg_term = (lambda/(2*m))* sum (theta(2:end).^2);
            // 

            // Create a new writer and write the values to disk
            
            double fooby = (lambda / (double)m) * Matrix.Sum(Elementwise.Pow(Theta, 2));
            Theta = Elementwise.Pow(Theta, 2); // Take the square of the Theta's
            double regterm = (lambda / (2 * (double) m)) * Matrix.Sum(Theta);
            return regterm;
         }

        static public double L1Regularization(double [,] Theta, Double lambda, long m)
        {
            int[] removeRow = { 0 };
            Theta = Matrix.Get(Theta, 1, Theta.Rows(), removeRow);
            Theta = Elementwise.Abs(Theta);
            double regterm = (lambda / (2 * (double) m)) * Matrix.Sum (Theta); 
            return regterm;
        }

        static public double CostFunc(double[,] inMatrix, double[,] Theta, double[,] y, double lambda)
        {
            // 
            // Computes the cost of using theta as the parameter for logistic regression to fit the data points in X and y
            // 
            long m = inMatrix.Rows();
            long cols = inMatrix.Columns();

            double regterm;

            // z = X * Theta
            // hypothesis = sigmoid (z)
            double[,] z, hypothesis;
            z = Matrix.Dot(inMatrix, Theta);


            hypothesis = MLFunctions.math.Sigmoid(z);

            regterm = MLFunctions.math.L2Regularization(Theta, lambda, m);


            //double [,] yminus = y.Multiply(1, y); // (1-y)
            double[,] yminus = y;
            yminus = Elementwise.Multiply(yminus, -1);
          
            double[,] term1 = Matrix.Dot (yminus, (Elementwise.Log(hypothesis))); // -y.* log(hypothesis)
            
            double[,] term2 = Matrix.Dot (1.Subtract(y),  Elementwise.Log(1.Subtract(hypothesis)));
            // term2 =  (1-y) .*log(1-hypothesis)
            
            double term3 = Matrix.Sum (term1.Subtract(term2));

            double J = ((1/(double)m)) * term3 + regterm;

            // J = (1/m) * sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))) + reg_term; 
            
            return J ;
             
            
        } 
    }

}