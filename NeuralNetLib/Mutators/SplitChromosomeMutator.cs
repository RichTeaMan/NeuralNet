using System;
using System.Collections.Generic;
using System.Linq;
using RichTea.NeuralNetLib.Serialisation;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Creates a new neural net by combining the weights of two parent neural nets.
    /// </summary>
    public class SplitChromosomeMutator : INeuralNetTwoParentMutator
    {
        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        /// <summary>
        /// Initialises split chromosome mutator.
        /// </summary>
        /// <param name="random">Random.</param>
        public SplitChromosomeMutator(Random random)
        {
            _random = random;
        }

        /// <summary>
        /// Initialises split chromosome mutator.
        /// </summary>
        public SplitChromosomeMutator() : this(new Random()) { }

        /// <summary>
        /// Creates a new neural net from two parents by combining their sequence of weights.
        /// </summary>
        /// <remarks>
        /// This mutator splits at a random point in the parents' weight sequence, and then
        /// joins the weights from the first parent to the second parent. These weights are then
        /// used to create a child neural net. Parents must have the same number of nodes and weights.
        /// </remarks>
        /// Parent nets must have the same number of inputs, outputs and nodes.
        /// <exception cref="ArgumentException">
        /// </exception>
        /// <param name="firstParentNet">First parent net.</param>
        /// <param name="secondParentNet">Second parent net.</param>
        /// <returns>Child parent net derived from the parents.</returns>
        public Net GenetateMutatedNeuralNet(Net firstParentNet, Net secondParentNet)
        {
            if (firstParentNet.InputCount != secondParentNet.InputCount)
            {
                throw new ArgumentException("Nets must have the same number of inputs.");
            }

            if (firstParentNet.OutputCount != secondParentNet.OutputCount)
            {
                throw new ArgumentException("Nets must have the same number of outputs.");
            }

            var firstParentNodes = firstParentNet.CreateSerialisedNet().NodeLayers.SelectMany(nl => nl.Nodes).ToArray();
            var secondParentNodes = secondParentNet.CreateSerialisedNet().NodeLayers.SelectMany(nl => nl.Nodes).ToArray();

            if (firstParentNodes.Length != secondParentNodes.Length)
            {
                throw new ArgumentException("Nets must have the same number of nodes.");
            }

            // random up a split point
            int split = _random.Next(1, firstParentNodes.Length);

            var childNodes = firstParentNodes.Take(split).ToList();
            childNodes.AddRange(secondParentNodes.Skip(split));
            var childNodeStack = new Stack<SerialisedNode>(childNodes.Reverse<SerialisedNode>());

            List<SerialisedNodeLayer> childNodeLayers = new List<SerialisedNodeLayer>();
            foreach (var layer in firstParentNet.NodeLayers)
            {
                List<SerialisedNode> nodesInLayer = new List<SerialisedNode>();

                foreach (var nodeCount in Enumerable.Range(0, layer.Nodes.Length))
                {
                    nodesInLayer.Add(childNodeStack.Pop());
                }

                var nodeLayer = new SerialisedNodeLayer
                {
                    Nodes = nodesInLayer.ToArray()
                };

                childNodeLayers.Add(nodeLayer);
            }
            var childSerialisedNet = new SerialisedNet
            {
                NodeLayers = childNodeLayers.ToArray()
            };

            var childNet = childSerialisedNet.CreateNet();
            return childNet;
        }
    }
}
