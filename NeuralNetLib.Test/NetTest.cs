
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class NetTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputTest()
        {
            var net = new Net(new Random(), -2, 2);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectOutputTest()
        {
            var net = new Net(new Random(), 2, -2);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void IncorrectInputCalculationTest()
        {
            var net = new Net(new Random(), 5, 5);

            net.Calculate(new double[2]);
        }

        [TestMethod]
        public void NetToStringTest()
        {
            var net = new Net(new Random(), 5, 5);
            net.ToString();
        }
    }
}
