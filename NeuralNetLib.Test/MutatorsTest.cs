using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Mutators;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class MutatorsTest
    {

        [TestMethod]
        public void OneParentMutatorsDoNotMutateSourceTest()
        {
            var random = new Random();

            var randomMutator = new RandomMutator(random);
            var splitChromosomeMutator = new SplitChromosomeMutator(random);
            var singularRandomNodeMutator = new SingularRandomNodeMutator(random);
            var weakestNodeMutator = new WeakestNodeMutator(random);
            var crossoverNodesMutator = new CrossoverNodesMutator(random);
            var randomParameterMutator = new RandomParameterMutator(random);

            var mutators = new List<INeuralNetOneParentMutator>() {
                randomMutator,
                singularRandomNodeMutator,
                weakestNodeMutator,
                randomParameterMutator,
            };

            var net = new Net(random, 3, 1);

            var serialNet = net.CreateSerialisedNet();

            foreach(var mutator in mutators)
            {
                var child = mutator.GenetateMutatedNeuralNet(net);

                Assert.AreEqual(serialNet, net.CreateSerialisedNet());
                Assert.AreNotEqual(serialNet, child.CreateSerialisedNet());
            }

        }

        [TestMethod]
        public void TwoParentMutatorsDoNotMutateSourceTest()
        {
            var random = new Random();

            var splitChromosomeMutator = new SplitChromosomeMutator(random);
            var crossoverNodesMutator = new CrossoverNodesMutator(random);

            var mutators = new List<INeuralNetTwoParentMutator>() {
                splitChromosomeMutator,
                crossoverNodesMutator,
            };

            var net1 = new Net(random, 3, 1);
            var serialNet1 = net1.CreateSerialisedNet();

            var net2 = new Net(random, 3, 1);
            var serialNet2 = net2.CreateSerialisedNet();

            foreach (var mutator in mutators)
            {
                var child = mutator.GenetateMutatedNeuralNet(net1, net2);

                Assert.AreEqual(serialNet1, net1.CreateSerialisedNet());
                Assert.AreEqual(serialNet2, net2.CreateSerialisedNet());
                Assert.AreNotEqual(serialNet1, child.CreateSerialisedNet());
                Assert.AreNotEqual(serialNet2, child.CreateSerialisedNet());
            }

        }
    }
}
