using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class NetFactoryTest
    {
        /// <summary>
        /// Class under test.
        /// </summary>
        private NetFactory netFactory;

        [TestInitialize]
        public void Setup()
        {
            netFactory = new NetFactory();
        }

        [TestMethod]
        public void CreateMutatedNetTest()
        {
            Random random = new Random(5);
            Net net = netFactory.GenerateRandomNet(3, 1, random);

            Net mutatednet = netFactory.CreateMutatedNet(net, random, 0.001);

            Assert.AreNotEqual(net.CreateSerialisedNet(), mutatednet.CreateSerialisedNet());

            
        }
    }
}
