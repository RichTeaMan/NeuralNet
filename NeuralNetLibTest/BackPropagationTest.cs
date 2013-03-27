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

            foreach (var dataSet in prop.DataSets)
            {
                double result = node.Calculate(dataSet.Inputs);
                Assert.IsTrue(
                    (dataSet.Outputs[0] == 1.0 && result > 0.8) ||
                (dataSet.Outputs[0] == 0.0 && result < 0.2), "LogicNodeOR failed to learn.");

            }

            Assert.IsTrue(SSE < 0.2, "LogicNodeOR SSE after {0} epochs is '{1}'", epoch, SSE);
        }

        [TestMethod]
        public void LogicNodeAND()
        {
            INode node = new Node(2);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0 });    // 0 | 1 = 0
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0 });    // 1 | 0 = 0
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 1000;
            double SSE = prop.Train(node, epoch);

            Assert.IsTrue(SSE < 0.2, "LogicNodeAND SSE after {0} epochs is '{1}'", epoch, SSE);
        }

        [TestMethod]
        public void LogicNodeXOR()
        {
            INode node = new Node(2);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 0 });    // 1 | 1 = 0

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 1000;
            double SSE = prop.Train(node, epoch);
            // this problem isn't possible for a single node, so check that if fails.
            Assert.IsTrue(SSE > 0.8, "LogicNodeXOR SSE after {0} epochs is '{1}'", epoch, SSE);
        }

        [TestMethod]
        public void LogicNodeLayerOR()
        {
            INodeLayer NodeLayer = new NodeLayer(2, 1);

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
            double SSE = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(SSE < 0.2, "LogicNodeOR SSE after {0} epochs is '{1}'", epoch, SSE);
        }

        [TestMethod]
        public void LogicNodeLayerAND()
        {
            INodeLayer NodeLayer = new NodeLayer(2, 1);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0 });    // 0 | 1 = 0
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0 });    // 1 | 0 = 0
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 1000;
            double SSE = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(SSE < 0.2, "LogicNodeAND SSE after {0} epochs is '{1}'", epoch, SSE);
        }

        [TestMethod]
        public void LogicNodeLayerANDOR()
        {
            INodeLayer NodeLayer = new NodeLayer(2, 2);

            BackPropagation prop = new BackPropagation(2, 2);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0, 0 });    // 0 | 0 = 00
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0, 1 });    // 0 | 1 = 01
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0, 1 });    // 1 | 0 = 01
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1, 1 });    // 1 | 1 = 11

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 1000;
            double SSE = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(SSE < 0.2, "LogicNodeANDOR SSE after {0} epochs is '{1}'", epoch, SSE);
        }
    }
}
