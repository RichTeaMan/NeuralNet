using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class BackPropagation
    {
        private List<DataSet> _dataSets = new List<DataSet>();
        public DataSet[] DataSets
        {
            get { return _dataSets.ToArray(); }
        }

        public int InputCount { get; private set; }

        public int OutputCount { get; private set; }

        public double LearningRate { get; set; }

        public BackPropagation(int Inputs, int Outputs)
        {
            InputCount = Inputs;
            OutputCount = Outputs;
            LearningRate = 0.5;
        }

        public void AddDataSet(double[] Inputs, double[] Outputs)
        {
            var dataSet = new DataSet(Inputs, Outputs);
            AddDataSet(dataSet);
        }

        public void AddDataSet(DataSet DataSet)
        {
            if (DataSet.InputCount == InputCount && DataSet.OutputCount == OutputCount)
                _dataSets.Add(DataSet);
            else
                throw new ArgumentException("The supplied DataSet has the incorrect number of data elements.");
        }

        public double Train(INode Node, int Epochs = 1000)
        {
            if (Epochs < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != Node.Inputs)
                throw new ArgumentException("The given INode does not have the same number of Inputs as the DataSets.");

            if (OutputCount != 1)
                throw new ArgumentException("An INode only supports 1 output.");

            for (int i = 0; i < Epochs; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    double error = 0;
                    double result = Node.Calculate(dataSet.Inputs, dataSet.Outputs.First(), ref error);
                    
                    // weight delta = learning rate * error * weight
                    double biasDelta = LearningRate * error;
                    Node.Bias += biasDelta;
                    for (int w = 0; w < Node.Inputs; w++)
                    {
                        double delta = biasDelta * Node.Weights[w];
                        Node.Weights[w] += delta;
                    }
                    
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                double result = Node.Calculate(dataSet.Inputs, dataSet.Outputs.First(), ref error);
                SSE += Math.Pow(error, 2);
            }
            return SSE;
        }

        public void Train(INodeLayer Node, int Epochs = 1000)
        {

        }

        public void Train(INet Node, int Epochs = 1000)
        {

        }

    }
}
