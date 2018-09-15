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

        public Net ResizeInputs(Net net, int inputNumber)
        {
            var serialNet = net.CreateSerialisedNet();

            var newLayers = new List<SerialisedNodeLayer>();

            int layerInputs = inputNumber;
            foreach (var layer in serialNet.NodeLayers)
            {
                var newNodes = new List<SerialisedNode>();
                foreach (var node in layer.Nodes)
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
