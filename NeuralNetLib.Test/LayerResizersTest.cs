using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Resizers;
using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class LayerResizersTest
    {

        [TestMethod]
        public void RandomLayerResizerIncreaseTest()
        {
            int netInput = 6;

            int oldLayers = 3;
            int newLayers = 5;

            var random = new Random();

            var randomlayerResizer = new RandomLayerResizer(random);

            var net = new Net(random, netInput, 1, oldLayers);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomlayerResizer.ResizeLayers(net, newLayers);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(netInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(newLayers + 2, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach (var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(netInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[netInput];
            resizedNet.Calculate(inputs);
        }

        [TestMethod]
        public void RandomLayerResizerDecreaseTest()
        {
            int netInput = 6;

            int oldLayers = 5;
            int newLayers = 1;

            var random = new Random();

            var randomlayerResizer = new RandomLayerResizer(random);

            var net = new Net(random, netInput, 1, oldLayers);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomlayerResizer.ResizeLayers(net, newLayers);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(netInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(newLayers + 2, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach (var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(netInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[netInput];
            resizedNet.Calculate(inputs);
        }

        [TestMethod]
        public void RandomLayerResizerNoResizeTest()
        {
            int netInput = 6;

            int oldLayers = 5;
            int newLayers = 5;

            var random = new Random();

            var randomlayerResizer = new RandomLayerResizer(random);

            var net = new Net(random, netInput, 1, oldLayers);

            var serialNet = net.CreateSerialisedNet();

            var resizedNet = randomlayerResizer.ResizeLayers(net, newLayers);


            // test original net hasn't been modified
            Assert.AreEqual(serialNet, net.CreateSerialisedNet());

            // test input count
            Assert.AreEqual(netInput, resizedNet.InputCount);

            // test output count
            Assert.AreEqual(net.OutputCount, resizedNet.OutputCount);

            // test layer count
            Assert.AreEqual(newLayers + 2, resizedNet.Layers);

            // check non output layers have correct number of nodes
            foreach (var layer in resizedNet.NodeLayers.Take(resizedNet.Layers - 1))
            {
                Assert.AreEqual(netInput, layer.Nodes.Count);
            }

            // checkout output layer has correct number of nodes
            Assert.AreEqual(net.NodeLayers.Last().Nodes.Count, resizedNet.NodeLayers.Last().Nodes.Count);

            // test a calcuation can happen. we don't care about result
            var inputs = new double[netInput];
            resizedNet.Calculate(inputs);
        }

    }
}
