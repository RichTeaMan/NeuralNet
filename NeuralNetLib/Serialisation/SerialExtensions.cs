using System;
using System.Collections.Generic;

namespace RichTea.NeuralNetLib.Serialisation
{
    /// <summary>
    /// Extensions for serialised nodes.
    /// </summary>
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

        /// <summary>
        /// Seeds weights for a serialised node with random.
        /// </summary>
        /// <param name="node">Node to be modified.</param>
        /// <param name="random">Random.</param>
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
