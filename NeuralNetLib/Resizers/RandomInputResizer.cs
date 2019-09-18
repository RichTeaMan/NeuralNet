using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Resizers
{
    /// <summary>
    /// Creates nets with the specified number of inputs. The weights for new inputs are seeded randomly.
    /// </summary>
    public class RandomInputResizer : IInputResizer
    {
        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        /// <summary>
        /// Initialises random input resizer from a random.
        /// </summary>
        /// <param name="random">Random.</param>
        public RandomInputResizer(Random random)
        {
            _random = random ?? throw new ArgumentNullException(nameof(random));
        }

        /// <summary>
        /// Initialises random input resizer.
        /// </summary>
        public RandomInputResizer() : this(new Random()) { }

        /// <summary>
        /// Resizes inputs by creating new weights. This will also expand hidden layers so all layers (except the output)
        /// so there are as many nodes as inputs.
        /// </summary>
        /// <param name="net">Source net.</param>
        /// <param name="inputNumber">Number of inputs the net should have.</param>
        /// <returns>Net</returns>
        public Net ResizeInputs(Net net, int inputNumber)
        {
            var serialNet = net.CreateSerialisedNet();

            var newLayers = new List<SerialisedNodeLayer>();

            int layerInputs = inputNumber;
            foreach (var layer in serialNet.NodeLayers)
            {
                var newNodes = new List<SerialisedNode>();
                foreach (var node in layer.Nodes.Take(layerInputs))
                {
                    var weights = new List<double>();
                    while (weights.Count < layerInputs)
                    {
                        double weight;
                        if (weights.Count < node.Weights.Count())
                        {
                            weight = node.Weights[weights.Count];
                        }
                        else
                        {
                            weight = _random.NextDouble();
                        }
                        weights.Add(weight);
                    }

                    var newNode = new SerialisedNode()
                    {
                        Weights = weights.ToArray(),
                        Bias = node.Bias
                    };

                    newNodes.Add(newNode);
                }

                // create new codes to match inputs except for output layer
                if (newNodes.Count < layerInputs && layer != serialNet.NodeLayers.Last())
                {
                    var extraNodes = Enumerable.Range(0, layerInputs - newNodes.Count).Select(i => new SigmoidNode(layerInputs, _random).CreateSerialisedNode()).ToList();
                    newNodes.AddRange(extraNodes);
                }

                var newLayer = new SerialisedNodeLayer()
                {
                    Nodes = newNodes.ToArray()
                };
                newLayers.Add(newLayer);
                layerInputs = newLayer.Nodes.Length;
            }

            var newSerialNet = new SerialisedNet()
            {
                NodeLayers = newLayers.ToArray()
            };

            var newNet = newSerialNet.CreateNet();

            return newNet;
        }
    }
}
