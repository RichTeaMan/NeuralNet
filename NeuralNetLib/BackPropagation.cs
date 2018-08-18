using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Back propagation learning algorithm for <see cref="Net"/>.
    /// </summary>
    public class BackPropagation
    {
        private List<DataSet> _dataSets = new List<DataSet>();

        /// <summary>
        /// Gets data sets to train with.
        /// </summary>
        public DataSet[] DataSets
        {
            get { return _dataSets.ToArray(); }
        }

        /// <summary>
        /// Gets input count of the net.
        /// </summary>
        public int InputCount { get; }

        /// <summary>
        /// Gets the output count of the net.
        /// </summary>
        public int OutputCount { get; }

        /// <summary>
        /// Gets or sets learning rate. This is how much a parameter is adjusted while training.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Initialises back propagation.
        /// </summary>
        /// <param name="inputCounts">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        public BackPropagation(int inputCounts, int outputCount)
        {
            InputCount = inputCounts;
            OutputCount = outputCount;
            LearningRate = 0.2;
        }

        /// <summary>
        /// Adds a data set from the given inputs and outputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="outputs">Outputs.</param>
        /// <exception cref="ArgumentException">Throws if there is an irregular number of elements.</exception>
        public void AddDataSet(double[] inputs, double[] outputs)
        {
            var dataSet = new DataSet(inputs, outputs);
            AddDataSet(dataSet);
        }

        /// <summary>
        /// Adds a data set.
        /// </summary>
        /// <param name="DataSet">Data set.</param>
        /// <exception cref="ArgumentException">Throws if the data set has an irregular number of elements.</exception>
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
            for (int w = 0; w < Node.InputCount; w++)
            {
                double delta = Delta * Inputs[w];
                Node.Weights[w] += delta;
            }
        }
        
        /// <summary>
        /// Trains the node for the given number of epochs. The node is modified. The SSE is returned.
        /// </summary>
        /// <param name="node">Node to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        /// <returns>Sum of the square of the errors (SSE).</returns>
        public double Train(Node node, int epochCount = 1000)
        {
            if (epochCount < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != node.InputCount)
                throw new ArgumentException("The given Node does not have the same number of Inputs as the DataSets.");

            if (OutputCount != 1)
                throw new ArgumentException("An Node only supports 1 output.");

            for (int i = 0; i < epochCount; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    double delta = 0;
                    double result = node.Calculate(dataSet.Inputs, dataSet.Outputs.First(), ref delta);
                    
                    // weight delta = learning rate * error * weight
                    AdjustNode(node, dataSet.Inputs, LearningRate * delta);
                    
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double result = node.Calculate(dataSet.Inputs);
                SSE += Math.Pow(dataSet.Outputs.First() - result, 2);
            }
            return SSE;
        }

        /// <summary>
        /// Trains the node layer for the given number of epochs. The node layer is modified. The SSE is returned.
        /// </summary>
        /// <param name="nodeLayer">Node layer to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        /// <returns>Sum of the square of the errors (SSE).</returns>
        public double Train(NodeLayer nodeLayer, int epochCount = 1000)
        {
            if (epochCount < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != nodeLayer.InputCount)
                throw new ArgumentException("The given NodeLayer does not have the same number of Inputs as the DataSets.");

            if (OutputCount != nodeLayer.OutputCount)
                throw new ArgumentException("The given NodeLayer does not have the same number of Outputs as the DataSets.");

            for (int i = 0; i < epochCount; i++)
            {
                for (int n = 0; n < nodeLayer.OutputCount; n++)
                {
                    foreach(var dataSet in DataSets)
                    {
                        double error = 0;
                        nodeLayer.Nodes[n].Calculate(dataSet.Inputs, dataSet.Outputs[n], ref error);

                        double delta = error * LearningRate;
                        AdjustNode(nodeLayer.Nodes[n], dataSet.Inputs, delta);
                    }
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                nodeLayer.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += Math.Pow(error, 2);
            }
            return SSE;
        }

        /// <summary>
        /// Trains net for the given number of epochs. The net is modified. The SSE is returned.
        /// </summary>
        /// <param name="net">Net to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        /// <returns>Sum of the square of the errors (SSE).</returns>
        public double Train(Net net, int epochCount = 1000)
        {
            var deltas = new Dictionary<Node, double>();
            foreach (var nodeLayer in net.NodeLayers)
            {
                foreach(var node in nodeLayer.Nodes)
                {
                    deltas.Add(node, 0.0);
                }
            }

            for (int i = 0; i < epochCount; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    double error = 0;
                    var results = net.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);

                    // set delta of output nodes
                    for(int r = 0; r < results.Length; r++)
                    {
                        double delta = (dataSet.Outputs[r] - results[r]) * results[r] * (1 - results[r]);
                        deltas[net.NodeLayers.Last().Nodes[r]] = delta;
                    }

                    for(int l = net.NodeLayers.Length - 2; l >= 0; l--)
                    {
                        for(int l2 = 0; l2 < net.NodeLayers[l].Nodes.Length; l2++)
                        {
                            var node = net.NodeLayers[l].Nodes[l2];
                            double delta = 0;
                            foreach(var linkedNode in net.NodeLayers[l+1].Nodes)
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
                var result = net.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += error;
            }
            return SSE;
        }

    }
}
