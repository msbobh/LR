using System;
using Accord.IO;
using Accord.Math;
using Functions;
using System.IO;
using strings;

namespace LogisticReg
{
    class Program
    {
        static void Main(string[] args)
        {
            // Initializations
            double alpha = .01; // Alpha values to try .001, .003, .01, .03, .1, .3, 1 
            double Lambda = 10;
            int iterations = 1500;
            string ThetaFile = "Thetasave.csv";

            if (args.Length < 2)
            {
                Console.WriteLine(mystrings.usage);
                System.Environment.Exit(-1);
            }
            if (!File.Exists(args[0]))
            {
                Console.WriteLine("Training file {0} not found!", args[0]);
                System.Environment.Exit(-1);
            }

            if (!File.Exists(args[1]))
            {
                Console.WriteLine("Label file {0} not found!", args[1]);
                System.Environment.Exit(-1);
            }

            if (args.Length > 2 && File.Exists(args[2]))
            {
                alpha = Convert.ToDouble(args[2]); ;

            }
            else
            {
                Console.WriteLine("Using default alpha {0}", alpha);
            }

            string trainingfile = args[0];
            string labelfile = args[1];
            // maybe add check for csv format??


            CsvReader training_samples = new CsvReader(trainingfile, false);
            //int cols = training_samples.Columns();
            double[,] input = training_samples.ToMatrix<double>();
            int rows = input.Rows();
            int cols = input.Columns();

            CsvReader labelscsv = new CsvReader(labelfile, false);
            double[,] labels = labelscsv.ToMatrix<double>();


            if ((labels.Rows() != input.Rows()))
            {
                Console.WriteLine(mystrings.SamplesDontMatch, labels.Length, input.Length);
            }

            Console.WriteLine("Training Set rows = {0}, Columns = {1}", rows, cols);
            Console.WriteLine("Label set rows = {0}, Columns = {1}", labels.Rows(), labels.Columns());

            // Initial guess for Theta is all zeros and n x 1 vector of zeros, where n is the number of features (columns)

            double[,] init_theta = Matrix.Zeros<double>(cols, 1);
            double[,] ones_theta = Matrix.Ones<double>(cols, 1);
            double[,] test_theta = Matrix.Random(cols, 1);


            Console.WriteLine(mystrings.running, iterations);
            /* 
             * Using Accord.net
             * 
             */

            double[,] theta = utilityfunctions.GradientDescent(input, labels, init_theta, alpha, iterations, Lambda);

            StreamWriter checkthis;
            try
            {
                checkthis = new StreamWriter("LearnedTheta.csv");
                checkthis.Close();
            }
            catch (IOException)
            {
                Console.WriteLine("Oopsie");
            }

            if (!utilityfunctions.FileOpen(ThetaFile))
            {
                Console.WriteLine(mystrings.File_in_use, ThetaFile);
                System.Environment.Exit(-1);
            }
            else
            {
                // Create a new writer and write the values to disk
                using (CsvWriter writer = new CsvWriter(ThetaFile))
                {
                    writer.Write(theta);
                } // closes at the end of the using block

            }

        }
    }


         
}
