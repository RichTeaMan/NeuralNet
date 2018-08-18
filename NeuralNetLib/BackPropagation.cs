using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
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
            LearningRate = 0.2;
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

        private void AdjustNode(Node Node, double[] Inputs, double Delta)
        {
            Node.Bias += Delta;
            for (int w = 0; w < Node.Inputs; w++)
            {
                double delta = Delta * Inputs[w];
                Node.Weights[w] += delta;
            }
        }
        
        public double Train(Node Node, int Epochs = 1000)
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
                    double delta = 0;
                    double result = Node.Calculate(dataSet.Inputs, dataSet.Outputs.First(), ref delta);
                    
                    // weight delta = learning rate * error * weight
                    AdjustNode(Node, dataSet.Inputs, LearningRate * delta);
                    
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double result = Node.Calculate(dataSet.Inputs);
                SSE += Math.Pow(dataSet.Outputs.First() - result, 2);
            }
            return SSE;
        }

        public double Train(NodeLayer NodeLayer, int Epochs = 1000)
        {
            if (Epochs < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != NodeLayer.Inputs)
                throw new ArgumentException("The given INodeLayer does not have the same number of Inputs as the DataSets.");

            if (OutputCount != NodeLayer.Outputs)
                throw new ArgumentException("The given INodeLayer does not have the same number of Outputs as the DataSets.");

            for (int i = 0; i < Epochs; i++)
            {
                for (int n = 0; n < NodeLayer.Outputs; n++)
                {
                    foreach(var dataSet in DataSets)
                    {
                        double error = 0;
                        NodeLayer.Nodes[n].Calculate(dataSet.Inputs, dataSet.Outputs[n], ref error);

                        double delta = error * LearningRate;
                        AdjustNode(NodeLayer.Nodes[n], dataSet.Inputs, delta);
                    }
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                NodeLayer.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += Math.Pow(error, 2);
            }
            return SSE;
        }

        public double Train(Net Net, int Epochs = 1000)
        {
            var deltas = new Dictionary<Node, double>();
            foreach (var nodeLayer in Net.NodeLayers)
            {
                foreach(var node in nodeLayer.Nodes)
                {
                    deltas.Add(node, 0.0);
                }
            }

            for (int i = 0; i < Epochs; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    double error = 0;
                    var results = Net.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);

                    // set delta of output nodes
                    for(int r = 0; r < results.Length; r++)
                    {
                        double delta = (dataSet.Outputs[r] - results[r]) * results[r] * (1 - results[r]);
                        deltas[Net.NodeLayers.Last().Nodes[r]] = delta;
                    }

                    for(int l = Net.NodeLayers.Length - 2; l >= 0; l--)
                    {
                        for(int l2 = 0; l2 < Net.NodeLayers[l].Nodes.Length; l2++)
                        {
                            var node = Net.NodeLayers[l].Nodes[l2];
                            double delta = 0;
                            foreach(var linkedNode in Net.NodeLayers[l+1].Nodes)
                            {
                                // add delta * weight of that node
                                delta += node.Result * (1 - node.Result) * linkedNode.Weights[l2] * deltas[linkedNode];
                                
                            }
                            // save delta for other nodes
                            deltas[node] = delta;

                        }
                    }
                    foreach (var delta in deltas)
                    {
                        AdjustNode(delta.Key, dataSet.Inputs, delta.Value);
                    }
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                var result = Net.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += error;
            }
            return SSE;
        }

    }
}
