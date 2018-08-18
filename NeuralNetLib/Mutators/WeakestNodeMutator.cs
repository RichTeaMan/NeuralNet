using RichTea.Common.Extensions;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Completey reseeds the weakest node in the net.
    /// </summary>
    public class WeakestNodeMutator : INeuralNetOneParentMutator
    {

        private Random _random;

        public WeakestNodeMutator(Random random)
        {
            _random = random;
        }

        public WeakestNodeMutator() : this(new Random()) { }

        public Net GenetateMutatedNeuralNet(Net parentNet)
        {
            Net mutatedNet = new Net(parentNet.InputCount, parentNet.OutputCount, parentNet.Layers);
            mutatedNet.SeedWeights(parentNet);

            var mutatedNode = mutatedNet.Nodes.OrderBy(n => n.Bias + n.Weights.Sum()).First();
            mutatedNode.SeedWeights(_random);
            return mutatedNet;
        }
    }
}
