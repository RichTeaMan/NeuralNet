using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class SerialisedNodeLayerTest
    {

        [TestMethod]
        public void DefaultNodeLayerEqualsTest()
        {
            var a = new NodeLayer(3, 1, new Random(5));
            var b = new NodeLayer(3, 1, new Random(5));

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void DefaultNodeLayerNotEqualsTest()
        {
            var a = new NodeLayer(3, 1, new Random(5));
            var b = new NodeLayer(2, 1, new Random(5));

            Assert.AreNotEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void SeededNodeLayerEqualsTest()
        {
            var randA = new Random(5);
            var randB = new Random(5);
            var a = new NodeLayer(3, 1, randA);
            var b = new NodeLayer(3, 1, randB);

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void NodeLayerSeedWeightsTest()
        {
            var rand = new Random(5);
            var a = new NodeLayer(3, 1,rand);
            var b = new NodeLayer(a.Nodes);

            Assert.AreEqual(a.CreateSerialisedNodeLayer(), b.CreateSerialisedNodeLayer());
        }

        [TestMethod]
        public void SeededNodeLayerNotEqualsTest()
        {
            var rand = new Random(5);
            var a = new NodeLayer(3, 1, rand);
            var b = new NodeLayer(3, 1, rand);

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
                Nodes = new[] { serialisedNodeA, serialisedNodeB }
            };

            var nodeLayer = serialisedNodeLayer.CreateNodeLayer();

            Assert.AreEqual(nodeABias, nodeLayer.Nodes[0].Bias);
            CollectionAssert.AreEquivalent(nodeAWeights, nodeLayer.Nodes[0].Weights.ToArray());

            Assert.AreEqual(nodeBBias, nodeLayer.Nodes[1].Bias);
            CollectionAssert.AreEquivalent(nodeBWeights, nodeLayer.Nodes[1].Weights.ToArray());

            Assert.AreEqual(2, nodeLayer.InputCount);
            Assert.AreEqual(2, nodeLayer.OutputCount);
        }
    }
}
