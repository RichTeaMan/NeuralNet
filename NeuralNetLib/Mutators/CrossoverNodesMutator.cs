using System;
using System.Collections.Generic;
using System.Linq;
using RichTea.NeuralNetLib.Serialisation;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Creates a net from two parents by selecting a single node in series from them.
    /// </summary>
    public class CrossoverNodesMutator : INeuralNetTwoParentMutator
    {
        private Random _random;

        public CrossoverNodesMutator(Random random)
        {
            _random = random;
        }

        public CrossoverNodesMutator() : this(new Random()) { }

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

            List<SerialisedNodeLayer> childNodeLayers = new List<SerialisedNodeLayer>();
            int nodeIndex = 0;
            foreach (var layer in firstParentNet.NodeLayers)
            {
                List<SerialisedNode> nodesInLayer = new List<SerialisedNode>();

                foreach (var nodeCount in Enumerable.Range(0, layer.Nodes.Length))
                {
                    SerialisedNode node;
                    if (_random.Next() % 2 == 0)
                    {
                        node = firstParentNodes[nodeIndex];
                    }
                    else
                    {
                        node = secondParentNodes[nodeIndex];
                    }

                    nodesInLayer.Add(node);
                    nodeIndex++;
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
