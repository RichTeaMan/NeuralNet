using System;
using System.Collections.Generic;
using System.Linq;
using RichTea.NeuralNetLib.Serialisation;

namespace RichTea.NeuralNetLib.Mutators
{
    public class SplitChromosomeMutator : INeuralNetTwoParentMutator
    {
        private Random _random;

        public SplitChromosomeMutator(Random random)
        {
            _random = random;
        }

        public SplitChromosomeMutator() : this(new Random()) { }

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
