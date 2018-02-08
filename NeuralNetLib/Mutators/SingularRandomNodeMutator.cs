using RichTea.Common.Extensions;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Completey reseeds one random node in the net.
    /// </summary>
    public class SingularRandomNodeMutator : INeuralNetOneParentMutator
    {

        private Random _random;

        public SingularRandomNodeMutator(Random random)
        {
            _random = random;
        }

        public SingularRandomNodeMutator() : this(new Random()) { }

        public Net GenetateMutatedNeuralNet(Net parentNet)
        {
            Net mutatedNet = new Net(parentNet.Inputs, parentNet.Outputs, parentNet.Layers);
            mutatedNet.SeedWeights(parentNet);

            int nodeIndexToMutate = _random.Next(parentNet.NodeCount);

            var mutatedNode = mutatedNet.Nodes.ElementAt(nodeIndexToMutate);
            mutatedNode.SeedWeights(_random);
            return mutatedNet;
        }
    }
}
