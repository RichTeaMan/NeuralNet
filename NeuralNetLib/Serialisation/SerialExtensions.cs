using System;
using System.Collections.Generic;

namespace RichTea.NeuralNetLib.Serialisation
{
    public static class SerialExtensions
    {
        /// <summary>
        /// Gets all nodes in the net.
        /// </summary>
        public static IEnumerable<SerialisedNode> Nodes(this SerialisedNet net)
        {
            foreach (var nodeLayer in net.NodeLayers)
            {
                foreach (var node in nodeLayer.Nodes)
                {
                    yield return node;
                }
            }
        }


        public static void SeedWeights(this SerialisedNode node, Random random)
        {
            node.Bias = random.NextDouble();
            for(int i = 0; i < node.Weights.Length; i++)
            {
                node.Weights[i] = random.NextDouble();
            }
        }
    }
}
