using System;
using System.IO;
using Accord.IO;
using Accord.Math;



namespace Functions
{
    class utilityfunctions
    {

        static bool IsFileinUse(FileInfo file)
        {
            FileStream stream = null;

            try
            {
                stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                //the file is unavailable because it is:
                //still being written to
                //or being processed by another thread
                //or does not exist (has already been processed)
                return true;
            }
            finally
            {
                if (stream != null)
                    stream.Close();
            }
            return false;
        }

        static public bool FileOpen(string chkfilename)
        {
            StreamWriter checkthis;
            try
            {
                checkthis = new StreamWriter(chkfilename);
                checkthis.Close();
            }
            catch (IOException)
            {
                return false;
            }
            return true;
        }

        static public float SumCalc(double [,] inMatrix)
        {
            float value = 0;
            for(int i = 0; i < inMatrix.Rows(); i++)
            {
                 value = value + (float)inMatrix[i,0];
            }
            return value;
        }
        static public bool WriteCSV(string CSVname, double[] jValues)
        {
         
            StreamWriter outfile = null;
            try
            {
                outfile = new StreamWriter(CSVname);
            }
            catch (IOException)
            {
                return false;
            }
            
            foreach (var row in jValues)
            {
                outfile.WriteLine(row.ToString());
            }

            outfile.Close();
            return true;
        }

        
        static public double [,] GradientDescent(double[,] X, double [,] y, double [,] Theta, double alpha, int iterations, double lambda)
        { 
            // Performs Gradient Descent to learn Theta, updates theta by
            // taking num_iters gradient steps with learning rate alpha
            //

            double[] J_history = new double[iterations];
            J_history.Initialize();
            double [,] error;
            string JValsFname = "JValues.csv"; // File to save the calculated error
            
            long m = X.Rows();
            for (int i = 0; i < iterations; i++)
            {
                // Perform a single gradient step
                // J = (X * Theta) - y;
                error = (Matrix.Dot(X,Theta)).Subtract(y);
                // theta = theta - ((alpha/m) * X'*error);
                Theta = Theta.Subtract(Elementwise.Multiply((alpha / (double)m), Matrix.Dot(X.Transpose(), error)));
                Console.Write('.');
                if (i % Console.WindowWidth - 1 == 0) Console.WriteLine();

                J_history[i] = MLFunctions.math.CostFunc(X, Theta, y, lambda); // Calculate error based on current Theta
                // Update theta based on magnitude of error
                
                
                
            }
            Console.WriteLine();
            //int x = 1;
            foreach (var cost in J_history)
            {
                Console.WriteLine("J = " + string.Format("{0:0.0000}", cost));
                //x++;
            } 
            if ((utilityfunctions.WriteCSV(JValsFname, J_history) == false))
            {
                Console.WriteLine(strings.mystrings.File_in_use, JValsFname);
                System.Environment.Exit(-1);
            }
              
            return Theta;
            
        } 
               
    } 
}

 