using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Mutators;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class SerialisedNetTest
    {

        [TestMethod]
        public void DefaultNetEqualsTest()
        {
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            Assert.AreEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
        }

        [TestMethod]
        public void DefaultNetHashCodeEqualsTest()
        {
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            Assert.AreEqual(a.CreateSerialisedNet().GetHashCode(), b.CreateSerialisedNet().GetHashCode());
        }

        [TestMethod]
        public void DefaultNetNotEqualsTest()
        {
            var a = new Net(3, 1);
            var b = new Net(2, 1);

            Assert.AreNotEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
        }

        [TestMethod]
        public void DefaultNetHashCodeNotEqualsTest()
        {
            var a = new Net(3, 1);
            var b = new Net(2, 1);

            Assert.AreNotEqual(a.CreateSerialisedNet().GetHashCode(), b.CreateSerialisedNet().GetHashCode());
        }

        [TestMethod]
        public void SeededNetEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            a.SeedWeights(randA);
            b.SeedWeights(randB);

            Assert.AreEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
        }

        [TestMethod]
        public void SeededNetNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            a.SeedWeights(rand);
            b.SeedWeights(rand);

            Assert.AreNotEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
        }

        [TestMethod]
        public void SeededNetHashCodeEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            a.SeedWeights(randA);
            b.SeedWeights(randB);

            Assert.AreEqual(a.CreateSerialisedNet().GetHashCode(), b.CreateSerialisedNet().GetHashCode());
        }

        [TestMethod]
        public void SeededNetHashCodeNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new Net(3, 1);
            var b = new Net(3, 1);

            a.SeedWeights(rand);
            b.SeedWeights(rand);

            Assert.AreNotEqual(a.CreateSerialisedNet().GetHashCode(), b.CreateSerialisedNet().GetHashCode());
        }

        [TestMethod]
        public void NetSeedWeightsTest()
        {
            var rand = new Random(5);
            var a = new Net(6, 1);
            var b = new Net(6, 1);

            a.SeedWeights(rand);
            b.SeedWeights(a);

            Assert.AreEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
        }

        [TestMethod]
        public void SerialisedNetDoesNotMutateSourceTest()
        {
            var rand = new Random(5);
            var net = new Net(3, 1);

            net.SeedWeights(rand);

            var firstNode = net.Nodes.First();
            double bias = firstNode.Bias;
            double weight = firstNode.Weights.First();

            var serialisedNet = net.CreateSerialisedNet();

            Assert.AreEqual(bias, serialisedNet.NodeLayers.First().Nodes.First().Bias);
            Assert.AreEqual(weight, serialisedNet.NodeLayers.First().Nodes.First().Weights[0]);

            // modify node
            serialisedNet.NodeLayers.First().Nodes.First().Bias += 0.1;
            serialisedNet.NodeLayers.First().Nodes.First().Weights[0] += 0.1;

            Assert.AreEqual(bias, firstNode.Bias);
            Assert.AreEqual(weight, firstNode.Weights.First());
        }

        [TestMethod]
        public void DeserialisedNetTest()
        {
            double[] node1AWeights = new[] { 0.2, 0.9 };
            double node1ABias = 0.125;
            var serialisedNode1A = new SerialisedNode
            {
                Weights = node1AWeights,
                Bias = node1ABias
            };

            double[] node1BWeights = new[] { 0.3, 0.8 };
            double node1BBias = 0.305;
            var serialisedNode1B = new SerialisedNode
            {
                Weights = node1BWeights,
                Bias = node1BBias
            };

            double[] node2AWeights = new[] { 0.4, 0.18 };
            double node2ABias = 0.750;
            var serialisedNode2A = new SerialisedNode
            {
                Weights = node2AWeights,
                Bias = node2ABias
            };

            double[] node2BWeights = new[] { 0.6, 0.16 };
            double node2BBias = 0.525;
            var serialisedNode2B = new SerialisedNode
            {
                Weights = node2BWeights,
                Bias = node2BBias
            };

            var serialisedNodeLayer1 = new SerialisedNodeLayer
            {
                Nodes = new[] { serialisedNode1A, serialisedNode1B }
            };

            var serialisedNodeLayer2 = new SerialisedNodeLayer
            {
                Nodes = new[] { serialisedNode2A, serialisedNode2B }
            };

            var serialisedNet = new SerialisedNet
            {
                NodeLayers = new [] { serialisedNodeLayer1, serialisedNodeLayer2 }
            };

            var Net = serialisedNet.CreateNet();

            Assert.AreEqual(node1ABias, Net.NodeLayers[0].Nodes[0].Bias);
            CollectionAssert.AreEquivalent(node1AWeights, Net.NodeLayers[0].Nodes[0].Weights);
            Assert.AreEqual(node1BBias, Net.NodeLayers[0].Nodes[1].Bias);
            CollectionAssert.AreEquivalent(node1BWeights, Net.NodeLayers[0].Nodes[1].Weights);

            Assert.AreEqual(node2ABias, Net.NodeLayers[1].Nodes[0].Bias);
            CollectionAssert.AreEquivalent(node2AWeights, Net.NodeLayers[1].Nodes[0].Weights);
            Assert.AreEqual(node2BBias, Net.NodeLayers[1].Nodes[1].Bias);
            CollectionAssert.AreEquivalent(node2BWeights, Net.NodeLayers[1].Nodes[1].Weights);

            Assert.AreEqual(2, Net.InputCount);
            Assert.AreEqual(2, Net.OutputCount);
        }

        [TestMethod]
        public void HashCodeTest()
        {
            int count = 10000;
            var hashList = new List<int>();
            Random random = new Random(12);

            // nets produced from this mutator sometimes have the same hashcode as their parent.
            var weakestNodeMutator = new WeakestNodeMutator(random);

            foreach(var i in Enumerable.Range(0, count))
            {
                Net net = new Net(5, 1, 3);
                net.SeedWeights(random);

                var serialNet = net.CreateSerialisedNet();
                hashList.Add(serialNet.GetHashCode());

                var mutatedNet = weakestNodeMutator.GenetateMutatedNeuralNet(net);

                var mutSerialNet = mutatedNet.CreateSerialisedNet();
                hashList.Add(mutSerialNet.GetHashCode());

                if (serialNet.GetHashCode() == mutSerialNet.GetHashCode())
                {
                    var s = new
                    {
                        a = serialNet.NodeLayers.Select(n => n.GetHashCode()).ToArray(),
                        b = mutSerialNet.NodeLayers.Select(n => n.GetHashCode()).ToArray()
                    };

                    for (int layerIndex = 0; layerIndex < serialNet.NodeLayers.Length; layerIndex++)
                    {
                        var oLayer = serialNet.NodeLayers[layerIndex];
                        var mLayer = mutSerialNet.NodeLayers[layerIndex];

                        if (oLayer != mLayer)
                        {

                            for (int nodeIndex = 0; nodeIndex < oLayer.Nodes.Length; nodeIndex++)
                            {
                                var oNode = oLayer.Nodes[nodeIndex];
                                var mNode = mLayer.Nodes[nodeIndex];

                                if (oNode != mNode)
                                {
                                    Debug.WriteLine("Different node in identical net hash found.");
                                }
                            }

                            Debug.WriteLine("Different net layer in identical net hash found.");
                        }
                    }

                    Debug.WriteLine("Identical net hash found.");
                }
            }

            int uniqueHashes = hashList.Distinct().Count();

            Assert.AreEqual(count * 2, uniqueHashes);
        }
    }
}
