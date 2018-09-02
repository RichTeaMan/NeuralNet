using RichTea.Common.Extensions;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Completely reseeds one random node in the net.
    /// </summary>
    public class SingularRandomNodeMutator : INeuralNetOneParentMutator
    {
        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        /// <summary>
        /// Initialises singular random node mutator.
        /// </summary>
        /// <param name="random">Random.</param>
        public SingularRandomNodeMutator(Random random)
        {
            _random = random;
        }

        /// <summary>
        /// Initialises singular random node mutator.
        /// </summary>
        public SingularRandomNodeMutator() : this(new Random()) { }

        /// <summary>
        /// Create a new neural net where a single weight has been reseeded.
        /// </summary>
        /// <param name="parentNet">Parent net.</param>
        /// <returns>Child neural net.</returns>
        public Net GenetateMutatedNeuralNet(Net parentNet)
        {
            Net mutatedNet = new Net(parentNet.InputCount, parentNet.OutputCount, parentNet.Layers);
            mutatedNet.SeedWeights(parentNet);

            int nodeIndexToMutate = _random.Next(parentNet.NodeCount);

            var mutatedNode = mutatedNet.Nodes.ElementAt(nodeIndexToMutate);
            mutatedNode.SeedWeights(_random);
            return mutatedNet;
        }
    }
}
