using RichTea.Common.Extensions;
using System;

namespace RichTea.NeuralNetLib.Mutators
{
    public class RandomMutator : INeuralNetOneParentMutator
    {
        public double Deviation { get; set; } = 0.001;

        private Random _random;

        public RandomMutator(Random random)
        {
            _random = random;
        }

        public RandomMutator() : this(new Random()) { }

        public Net GenetateMutatedNeuralNet(Net parentNet)
        {
            Net mutatedNet = new Net(parentNet.InputCount, parentNet.OutputCount, parentNet.Layers);
            mutatedNet.SeedWeights(parentNet);

            foreach (var nodeLayer in mutatedNet.NodeLayers)
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
            return mutatedNet;
        }
    }
}
