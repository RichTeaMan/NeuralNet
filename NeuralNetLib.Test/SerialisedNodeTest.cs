using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class SerialisedNodeTest
    {

        [TestMethod]
        public void DefaultNodeEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new SigmoidNode(3, randA);
            var b = new SigmoidNode(3, randB);

            Assert.AreEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void DefaultNodeNotEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new SigmoidNode(3, randA);
            var b = new SigmoidNode(2, randB);

            Assert.AreNotEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());
        }

        [TestMethod]
        public void SeededNodeEqualsTest()
        {

            var randA = new Random(5);
            var randB = new Random(5);
            var a = new SigmoidNode(3,randA);
            var b = new SigmoidNode(3,randB);

            Assert.AreEqual(a.CreateSerialisedNode(), b.CreateSerialisedNode());

            // Ensure weight arrays eqivalent but not the same reference.
            CollectionAssert.AreEquivalent(a.Weights.ToArray(), b.Weights.ToArray());
            ReferenceEquals(a.Weights, b.Weights);
        }

        [TestMethod]
        public void SeededNodeNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new SigmoidNode(3, rand);
            var b = new SigmoidNode(3, rand);

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
            Assert.AreEqual(weights.Length, node.InputCount);
            CollectionAssert.AreEquivalent(weights, node.Weights.ToArray());
        }
    }
}
