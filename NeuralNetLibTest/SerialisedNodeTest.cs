using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLib;
using NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetLibTest
{
    [TestClass]
    public class SerialisedNodeTest
    {

        [TestMethod]
        public void DefaultNodeEqualsTest()
        {
            var a = new Node(3);
            var b = new Node(3);

            Assert.AreEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void DefaultNodeNotEqualsTest()
        {
            var a = new Node(3);
            var b = new Node(2);

            Assert.AreNotEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void SeededNodeEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new Node(3);
            var b = new Node(3);

            a.SeedWeights(randA);
            b.SeedWeights(randB);

            Assert.AreEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void SeededNodeNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new Node(3);
            var b = new Node(3);

            a.SeedWeights(rand);
            b.SeedWeights(rand);

            Assert.AreNotEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void DeserialisedNodeTest()
        {
            double[] weights = new[] { 0.2, 0.9 };
            double bias = 0.125;
            var serialisedNode = new SerialisedNode
            {
                Weights = weights,
                Bias = bias
            };

            var node = serialisedNode.CreateNode();

            Assert.AreEqual(bias, node.Bias);
            Assert.AreEqual(weights.Length, node.Inputs);
            CollectionAssert.AreEquivalent(weights, node.Weights);
        }
    }
}
