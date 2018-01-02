using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLib;
using NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetLibTest
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
        public void DefaultNetNotEqualsTest()
        {
            var a = new Net(3, 1);
            var b = new Net(2, 1);

            Assert.AreNotEqual(a.CreateSerialisedNet(), b.CreateSerialisedNet());
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

            Assert.AreEqual(2, Net.Inputs);
            Assert.AreEqual(2, Net.Outputs);
        }
    }
}
