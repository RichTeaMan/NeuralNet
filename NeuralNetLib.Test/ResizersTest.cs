using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Resizers;
using System;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class ResizersTest
    {

        [TestMethod]
        public void RandomInputResizerIncreaseTest()
        {
            int oldNetInput = 3;
            int newNetInput = 6;

            var random = new Random();

            var randomInputResizer = new RandomInputResizer(random);

            var net = new Net(oldNetInput, 1);
            net.SeedWeights(random);

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

            var net = new Net(oldNetInput, 1);
            net.SeedWeights(random);

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

            // test a calcuation can happen. we don't care about result
            var inputs = new double[newNetInput];
            resizedNet.Calculate(inputs);
        }

    }
}
