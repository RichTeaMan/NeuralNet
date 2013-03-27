using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetLib;

namespace NeuralNetLibTest
{
    [TestClass]
    public class BackPropagationTest
    {
        [TestMethod]
        public void LogicNodeOR()
        {
            INode node = new Node(2);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 1000;
            double SSE = prop.Train(node, epoch);

            Assert.IsTrue(SSE < 0.2, "LogicNodeOR SSE after {0} epochs is '{1}'", epoch, SSE);
        }
    }
}
