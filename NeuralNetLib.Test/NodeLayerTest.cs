using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class NodeLayerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputTest()
        {
            var nodeLayer = new NodeLayer(-2, 2, new Random());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectOutputTest()
        {
            var nodeLayer = new NodeLayer(2, -2, new Random());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputCalculationTest()
        {
            var nodeLayer = new NodeLayer(5, 5, new Random());

            nodeLayer.Calculate(new double[2]);
        }

        [TestMethod]
        public void NodeToStringTest()
        {
            var nodeLayer = new NodeLayer(5, 5, new Random());
            nodeLayer.ToString();
        }
    }
}
