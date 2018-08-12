using RichTea.Common.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Mutates the given amount of parameters from ParameterAmount by the given
    /// deviation.
    /// </summary>
    public class RandomParameterMutator : INeuralNetOneParentMutator
    {
        /// <summary>
        /// Gets or sets how much a parameter can mutate.
        /// </summary>
        public double Deviation { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets number of parameters to mutate.
        /// </summary>
        public int ParameterAmount { get; set; } = 1;

        private Random _random;

        public RandomParameterMutator(Random random)
        {
            _random = random;
        }

        public RandomParameterMutator() : this(new Random()) { }

        public Net GenetateMutatedNeuralNet(Net parentNet)
        {
            var serialisedParent = parentNet.CreateSerialisedNet();

            // get random index to mutate
            int weightCount = serialisedParent.NodeLayers.Sum(nl => nl.Nodes.Sum(n => n.Weights.Length + 1));

            var indices = new HashSet<int>(Enumerable.Range(0, ParameterAmount + 1).Select(i => _random.Next(weightCount)));

            int index = 0;
            foreach (var nodeLayer in serialisedParent.NodeLayers)
            {
                foreach (var node in nodeLayer.Nodes)
                {
                    if (indices.Contains(index))
                    {
                        var bias = node.Bias + _random.NextDoubleInRange(Deviation);
                        node.Bias = bias;
                    }
                    index++;
                    for (int i = 0; i < node.Weights.Length; i++)
                    {
                        if (indices.Contains(index))
                        {
                            var weight = node.Weights[i] + _random.NextDoubleInRange(Deviation);
                            node.Weights[i] = weight;
                        }
                        index++;
                    }
                }
            }

            var mutatedNet = serialisedParent.CreateNet();
            if (mutatedNet == parentNet)
            {
                throw new Exception("Inbred nets.");
            }
            return mutatedNet;
        }
    }
}
