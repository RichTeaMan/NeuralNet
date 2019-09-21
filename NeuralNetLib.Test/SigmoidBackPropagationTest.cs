using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class SigmoidBackPropagationTest
    {
        [TestMethod]
        public void LogicNodeOR()
        {
            Node node = new SigmoidNode(2, new Random());

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
            var backPropResult = prop.Train(node, epoch);

            foreach (var dataSet in prop.DataSets)
            {
                double result = backPropResult.Net.Calculate(dataSet.Inputs);
                Assert.IsTrue(
                    (dataSet.Outputs[0] == 1.0 && result > 0.8) ||
                (dataSet.Outputs[0] == 0.0 && result < 0.2), "LogicNodeOR failed to learn.");

            }

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNodeOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNodeAND()
        {
            Node node = new SigmoidNode(2, new Random());

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
            var backPropResult = prop.Train(node, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNodeAND SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNodeXOR()
        {
            Node node = new SigmoidNode(2, new Random());

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
            var backPropResult = prop.Train(node, epoch);
            // this problem isn't possible for a single node, so check that if fails.
            Assert.IsTrue(backPropResult.SSE > 0.8, "LogicNodeXOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNodeLayerOR()
        {
            NodeLayer NodeLayer = new NodeLayer(2, 1, new Random());

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
            var backPropResult = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNodeOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNodeLayerAND()
        {
            NodeLayer NodeLayer = new NodeLayer(2, 1, new Random());

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
            var backPropResult = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNodeAND SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNodeLayerANDOR()
        {
            NodeLayer NodeLayer = new NodeLayer(2, 2, new Random());

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
            var backPropResult = prop.Train(NodeLayer, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNodeANDOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNetOR()
        {
            Net Net = new Net(new Random(), 2, 1);

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
            var backPropResult = prop.Train(Net, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNetAND()
        {
            Net Net = new Net(new Random(), 2, 1);

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
            var backPropResult = prop.Train(Net, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetAND SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNetXOR()
        {
            Net Net = new Net(new Random(5), 2, 1);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 0 });    // 1 | 1 = 0

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 10000;
            var backPropResult = prop.Train(Net, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetXOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }

        [TestMethod]
        public void LogicNetANDORXOR()
        {
            Net Net = new Net(new Random(), 2, 3);

            BackPropagation prop = new BackPropagation(2, 3);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0, 0, 0 });    // 0 | 0 = 000
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0, 1, 1 });    // 0 | 1 = 011
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0, 1, 1 });    // 1 | 0 = 011
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1, 1, 0 });    // 1 | 1 = 110

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            int epoch = 10000;
            var backPropResult = prop.Train(Net, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetANDORXOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
        }
    }
}
