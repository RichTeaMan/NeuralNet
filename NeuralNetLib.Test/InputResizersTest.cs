using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Resizers;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class InputResizersTest
    {

        [TestMethod]
        public void RandomInputResizerIncreaseTest()
        {
            int oldNetInput = 3;
            int newNetInput = 6;

            var random = new Random();

            var randomInputResizer = new RandomInputResizer(random);

            var net = new Net(random, oldNetInput, 1);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomInputResizer.ResizeInputs(net, newNetInput);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(newNetInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(net.Layers, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach (var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(newNetInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[newNetInput];
            resizedNet.Calculate(inputs);
        }

        [TestMethod]
        public void RandomInputResizerDecreaseTest()
        {
            int oldNetInput = 6;
            int newNetInput = 3;

            var random = new Random();

            var randomInputResizer = new RandomInputResizer(random);

            var net = new Net(random, oldNetInput, 1);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomInputResizer.ResizeInputs(net, newNetInput);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(newNetInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(net.Layers, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach(var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(newNetInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[newNetInput];
            resizedNet.Calculate(inputs);
        }

        [TestMethod]
        public void RandomInputResizerNoResizeTest()
        {
            int oldNetInput = 4;
            int newNetInput = 4;

            var random = new Random();

            var randomInputResizer = new RandomInputResizer(random);

            var net = new Net(random, oldNetInput, 1);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomInputResizer.ResizeInputs(net, newNetInput);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(newNetInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(net.Layers, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach (var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(newNetInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[newNetInput];
            resizedNet.Calculate(inputs);

            Assert.AreEqual(serialNet, resizedNet.CreateSerialisedNet());
        }

    }
}
