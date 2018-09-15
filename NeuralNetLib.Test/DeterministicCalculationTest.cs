using Microsoft.VisualStudio.TestTools.UnitTesting;
using RichTea.NeuralNetLib.Mutators;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Test
{
    [TestClass]
    public class DeterministicCalculationTest
    {

        [TestMethod]
        public void DeterministicNetTest()
        {
            Net Net = new Net(new Random(), 2, 1);

            DataSet dataset = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1

            List<double> results = new List<double>();
            foreach (var i in Enumerable.Range(0, 1000)) {
                results.Add(Net.Calculate(dataset.Inputs).First());
            }

            Assert.AreEqual(1, results.Distinct().Count());
        }

        [TestMethod]
        public void MutatedDeterministicNetTest()
        {
            Net Net = new Net(new Random(), 2, 1);

            DataSet dataset = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1

            var mutator = new RandomParameterMutator();

            List<double> results = new List<double>();
            foreach (var i in Enumerable.Range(0, 1000))
            {
                var mutatedNet = mutator.GenetateMutatedNeuralNet(Net);
                double result1 = mutatedNet.Calculate(dataset.Inputs).First();
                double result2 = mutatedNet.Calculate(dataset.Inputs).First();
                Assert.AreEqual(result1, result2);
            }

        }

        [TestMethod]
        public void DeterministicDatasetEvaluatorTest()
        {
            Net Net = new Net(new Random(), 2, 1);

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1
            var dataSets = new[] { _1, _2, _3, _4 };

            foreach (var i in Enumerable.Range(0, 1000))
            {
                List<double> results = new List<double>();

                Net net = new Net(new Random(), 2, 1);
                double sse1 = 0;
                foreach (var dataSet in dataSets)
                {
                    double result = net.Calculate(dataSet.Inputs).First();
                    double error = (dataSet.Outputs.First() - result) * 100;
                    sse1 += Math.Pow(error, 2.0);
                }
                results.Add(sse1);
                double sse2 = 0;
                foreach (var dataSet in dataSets)
                {
                    double result = net.Calculate(dataSet.Inputs).First();
                    double error = (dataSet.Outputs.First() - result) * 100;
                    sse2 += Math.Pow(error, 2.0);
                }
                results.Add(sse2);
                int d = results.Distinct().Count();
                Assert.AreEqual(1, results.Distinct().Count());
            }

        }

    }
}
