﻿using RichTea.Common.Extensions;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Completey reseeds the weakest node in the net.
    /// </summary>
    public class WeakestNodeMutator : INeuralNetOneParentMutator
    {
        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        /// <summary>
        /// Initialises weakest node mutator.
        /// </summary>
        /// <param name="random">Ranodm.</param>
        public WeakestNodeMutator(Random random)
        {
            _random = random;
        }

        /// <summary>
        /// Initialises weakest node mutator.
        /// </summary>
        public WeakestNodeMutator() : this(new Random()) { }

        /// <summary>
        /// Creates a new child neural net from a parent net by reseeding the weakest node. The weakest node is determined by which has the smallest weights.
        /// </summary>
        /// <param name="parentNet">Parent neural net.</param>
        /// <returns>Child neural net.</returns>
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
