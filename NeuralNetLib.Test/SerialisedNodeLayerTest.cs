using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Serialisation;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class SerialisedNodeLayerTest
    {

        [TestMethod]
        public void DefaultNodeLayerEqualsTest()
        {
            var a = new NodeLayer(3, 1);
            var b = new NodeLayer(3, 1);

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void DefaultNodeLayerNotEqualsTest()
        {
            var a = new NodeLayer(3, 1);
            var b = new NodeLayer(2, 1);

            Assert.AreNotEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void SeededNodeLayerEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new NodeLayer(3, 1);
            var b = new NodeLayer(3, 1);

            a.SeedWeights(randA);
            b.SeedWeights(randB);

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void NodeLayerSeedWeightsTest()
        {
            var rand = new Random(5);
            var a = new NodeLayer(3, 1);
            var b = new NodeLayer(3, 1);

            a.SeedWeights(rand);
            b.SeedWeights(a);

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void SeededNodeLayerNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new NodeLayer(3, 1);
            var b = new NodeLayer(3, 1);

            a.SeedWeights(rand);
            b.SeedWeights(rand);

            Assert.AreNotEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void DeserialisedNodeLayerTest()
        {
            double[] nodeAWeights = new[] { 0.2, 0.9 };
            double nodeABias = 0.125;
            var serialisedNodeA = new SerialisedNode
            {
                Weights = nodeAWeights,
                Bias = nodeABias
            };

            double[] nodeBWeights = new[] { 0.3, 0.8 };
            double nodeBBias = 0.525;
            var serialisedNodeB = new SerialisedNode
            {
                Weights = nodeBWeights,
                Bias = nodeBBias
            };

            var serialisedNodeLayer = new SerialisedNodeLayer
            {
                Nodes = new [] {serialisedNodeA, serialisedNodeB}
            };

            var nodeLayer = serialisedNodeLayer.CreateNodeLayer();

            Assert.AreEqual(nodeABias, nodeLayer.Nodes[0].Bias);
            CollectionAssert.AreEquivalent(nodeAWeights, nodeLayer.Nodes[0].Weights);

            Assert.AreEqual(nodeBBias, nodeLayer.Nodes[1].Bias);
            CollectionAssert.AreEquivalent(nodeBWeights, nodeLayer.Nodes[1].Weights);

            Assert.AreEqual(2, nodeLayer.Inputs);
            Assert.AreEqual(2, nodeLayer.Outputs);
        }
    }
}
