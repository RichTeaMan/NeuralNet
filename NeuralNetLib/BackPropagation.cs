using RichTea.NeuralNetLib.Serialisation;
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

        private Node AdjustNode(Node node, double[] Inputs, double Delta)
        {
            var serialNode = node.CreateSerialisedNode();
            serialNode.Bias += Delta;
            for (int w = 0; w < serialNode.Weights.Length; w++)
            {
                double delta = Delta * Inputs[w];
                serialNode.Weights[w] += delta;
            }

            return serialNode.CreateNode();
        }
        
        /// <summary>
        /// Trains the node for the given number of epochs. A new node is returned in the result.
        /// </summary>
        /// <param name="node">Node to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        public BackPropagationResult<Node> Train(Node node, int epochCount = 1000)
        {
            if (epochCount < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != node.InputCount)
                throw new ArgumentException("The given Node does not have the same number of Inputs as the DataSets.");

            if (OutputCount != 1)
                throw new ArgumentException("An Node only supports 1 output.");

            var epochNode = node;
            for (int i = 0; i < epochCount; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    double delta = 0;
                    double result = epochNode.Calculate(dataSet.Inputs, dataSet.Outputs.First(), ref delta);
                    
                    // weight delta = learning rate * error * weight
                    epochNode = AdjustNode(epochNode, dataSet.Inputs, LearningRate * delta);
                    
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double result = epochNode.Calculate(dataSet.Inputs);
                SSE += Math.Pow(dataSet.Outputs.First() - result, 2);
            }
            var backPropagationResult = new BackPropagationResult<Node>(epochNode, SSE);
            return backPropagationResult;
        }

        /// <summary>
        /// Trains the node layer for the given number of epochs. A new node layer is returned in the result.
        /// </summary>
        /// <param name="nodeLayer">Node layer to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        public BackPropagationResult<NodeLayer> Train(NodeLayer nodeLayer, int epochCount = 1000)
        {
            if (epochCount < 1)
                throw new ArgumentException("At least 1 epoch is required.");

            if (DataSets.Length < 1)
                throw new ArgumentException("No DataSets have been loaded.");

            if (InputCount != nodeLayer.InputCount)
                throw new ArgumentException("The given NodeLayer does not have the same number of Inputs as the DataSets.");

            if (OutputCount != nodeLayer.OutputCount)
                throw new ArgumentException("The given NodeLayer does not have the same number of Outputs as the DataSets.");

            var epochNodeLayer = nodeLayer;
            for (int i = 0; i < epochCount; i++)
            {
                for (int n = 0; n < epochNodeLayer.OutputCount; n++)
                {
                    foreach(var dataSet in DataSets)
                    {
                        double error = 0;
                        epochNodeLayer.Nodes[n].Calculate(dataSet.Inputs, dataSet.Outputs[n], ref error);

                        double delta = error * LearningRate;

                        var newNodes = epochNodeLayer.Nodes.Select(node => AdjustNode(node, dataSet.Inputs, delta)).ToList();

                        epochNodeLayer = new NodeLayer(newNodes);
                    }
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                epochNodeLayer.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += Math.Pow(error, 2);
            }
            var backPropagationResult = new BackPropagationResult<NodeLayer>(epochNodeLayer, SSE);
            return backPropagationResult;
        }

        /// <summary>
        /// Trains net for the given number of epochs. A new net is returned in the result.
        /// </summary>
        /// <param name="net">Net to train.</param>
        /// <param name="epochCount">Number of epochs to train for.</param>
        /// <returns>Sum of the square of the errors (SSE).</returns>
        public BackPropagationResult<Net> Train(Net net, int epochCount = 1000)
        {
            var epochNet = net;
            for (int i = 0; i < epochCount; i++)
            {
                foreach (var dataSet in DataSets)
                {
                    var deltas = new Dictionary<SerialisedNode, double>();
                    foreach (var nodeLayer in net.NodeLayers)
                    {
                        foreach (var node in nodeLayer.Nodes)
                        {
                            deltas.Add(node.CreateSerialisedNode(), 0.0);
                        }
                    }

                    double error = 0;
                    var results = epochNet.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);

                    // set delta of output nodes
                    for(int r = 0; r < results.Length; r++)
                    {
                        double delta = (dataSet.Outputs[r] - results[r]) * results[r] * (1 - results[r]);
                        deltas[epochNet.NodeLayers.Last().Nodes[r].CreateSerialisedNode()] = delta;
                    }

                    for(int l = epochNet.NodeLayers.Count - 2; l >= 0; l--)
                    {
                        for(int l2 = 0; l2 < epochNet.NodeLayers[l].Nodes.Count; l2++)
                        {
                            var node = epochNet.NodeLayers[l].Nodes[l2];
                            double delta = 0;
                            foreach(var linkedNode in epochNet.NodeLayers[l+1].Nodes)
                            {
                                // add delta * weight of that node
                                delta += node.Result * (1 - node.Result) * linkedNode.Weights[l2] * deltas[linkedNode.CreateSerialisedNode()];
                                
                            }
                            // save delta for other nodes
                            deltas[node.CreateSerialisedNode()] = delta;
                        }
                    }

                    var nodes = epochNet.Nodes.ToArray();

                    var serialNet = epochNet.CreateSerialisedNet();
                    foreach(var serialLayer in serialNet.NodeLayers)
                    {
                        foreach(var serialNode in serialLayer.Nodes)
                        {
                            var delta = deltas[serialNode];
                            var node = serialNode.CreateNode();
                            var newNode = AdjustNode(node, dataSet.Inputs, delta);

                            serialNode.Bias = newNode.Bias;
                            for(int wI = 0; wI < serialNode.Weights.Length; wI++)
                            {
                                serialNode.Weights[wI] = newNode.Weights[wI];
                            }
                        }
                    }

                    epochNet = serialNet.CreateNet();
                }
            }

            double SSE = 0;
            foreach (var dataSet in DataSets)
            {
                double error = 0;
                var result = epochNet.Calculate(dataSet.Inputs, dataSet.Outputs, ref error);
                SSE += error;
            }
            var backPropagationResult = new BackPropagationResult<Net>(epochNet, SSE);
            return backPropagationResult;
        }

    }
}
