using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class NodeTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputTest()
        {
            var node = new SigmoidNode(-2, new Random());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputCalculationTest()
        {
            var node = new SigmoidNode(5, new Random());

            node.Calculate(new double[2]);
        }

        [TestMethod]
        public void NodeToStringTest()
        {
            var node = new SigmoidNode(5, new Random());
            node.ToString();
        }
    }
}
