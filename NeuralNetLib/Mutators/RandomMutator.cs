using RichTea.Common.Extensions;
using System;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Creates a neural net based upon a snigle parent where all weights are mutated by the deviation.
    /// </summary>
    public class RandomMutator : INeuralNetOneParentMutator
    {
        /// <summary>
        /// Gets or sets the mutation deviation.
        /// </summary>
        /// <remarks>
        /// Weights will be mutated by a random number between - and + deviation.
        /// </remarks>
        public double Deviation { get; set; } = 0.001;

        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        /// <summary>
        /// Initialises random mutator.
        /// </summary>
        /// <param name="random">Random.</param>
        public RandomMutator(Random random)
        {
            _random = random;
        }

        /// <summary>
        /// Initialises random mutator.
        /// </summary>
        public RandomMutator() : this(new Random()) { }

        /// <summary>
        /// Generates a new neural net with weights randomly changed from the parent. Deviation controls how the range of mutated values.
        /// </summary>
        /// <param name="parentNet">Parent net.</param>
        /// <returns>Child neural net.</returns>
        public Net GenetateMutatedNeuralNet(Net parentNet)
        {

            var serialisedNet = parentNet.CreateSerialisedNet();

            foreach (var nodeLayer in serialisedNet.NodeLayers)
            {
                foreach (var node in nodeLayer.Nodes)
                {
                    node.Bias += _random.NextDoubleInRange(Deviation);
                    for (int i = 0; i < node.Weights.Length; i++)
                    {
                        node.Weights[i] += _random.NextDoubleInRange(Deviation);
                    }
                }
            }
            var mutatedNet = serialisedNet.CreateNet();
            return mutatedNet;
        }
    }
}
