using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class NormaliserNodeTest
    {

        [TestMethod]
        public void PositiveUpperBoundNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(0);
            double result = node.Calculate(2);

            Assert.AreEqual(1.0, result);
        }

        [TestMethod]
        public void PositiveLowerBoundNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(0);
            node.Calculate(2);

            double result = node.Calculate(0);

            Assert.AreEqual(0.0, result);
        }

        [TestMethod]
        public void PositiveUpperBoundNonZeroNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(100);
            double result = node.Calculate(102);

            Assert.AreEqual(1.0, result);
        }

        [TestMethod]
        public void PositiveLowerBoundNonZeroNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(100);
            node.Calculate(102);

            double result = node.Calculate(100);

            Assert.AreEqual(0.0, result);
        }

        [TestMethod]
        public void NegativeUpperBoundNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(-2);
            double result = node.Calculate(0);

            Assert.AreEqual(1.0, result);
        }
        [TestMethod]
        public void NegativeLowerBoundNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(-2);
            node.Calculate(0);

            double result = node.Calculate(-2);

            Assert.AreEqual(0.0, result);
        }

        [TestMethod]
        public void NegativeUpperBoundNonZeroNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(-102);
            double result = node.Calculate(-100);

            Assert.AreEqual(1.0, result, double.Epsilon);
        }
        [TestMethod]
        public void NegativeLowerBoundNonZeroNormaliserNodeTest()
        {
            var node = new NormaliserNode();

            node.Calculate(-102);
            node.Calculate(-100);

            double result = node.Calculate(-102);

            Assert.AreEqual(0.0, result, double.Epsilon);
        }
    }
}
