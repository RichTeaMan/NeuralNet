using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace RichTea.NeuralNetLib.Test
{

    [Ignore]
    [TestClass]
    public class GeneticAlgorithmTrainerTest
    {

        class DatasetEvaluator : IFitnessEvaluator
        {
            public IReadOnlyList<DataSet> DataSetList { get; private set; }

            public ConcurrentBag<int> scores = new ConcurrentBag<int>();

            public DatasetEvaluator(IEnumerable<DataSet> dataSetList)
            {
                DataSetList = dataSetList.ToList() ?? throw new ArgumentNullException(nameof(dataSetList));
            }

            public int EvaluateNet(IReadOnlyList<Net> competingNets, Net evaluatingNet, TrainingStatus trainingStatus)
            {
                double sse = 0;
                foreach (var dataSet in DataSetList)
                {
                    double result = evaluatingNet.Calculate(dataSet.Inputs).First();
                    double error = (dataSet.Outputs.First() - result) * 100;
                    sse += Math.Pow(error, 2.0);
                }

                // convert to integer where int.max is most fit.
                int fitness = int.MaxValue;
                if (sse > 0)
                {
                    var f = int.MaxValue / sse;
                    if (f > int.MaxValue)
                    {
                        f = int.MaxValue;
                    }
                    else if (f < int.MinValue)
                    {
                        f = int.MinValue;
                    }
                    fitness = (int)Math.Round(f);
                }
                scores.Add(fitness);
                return fitness;
            }
        }

        [TestMethod]
        public void DatasetEvaluatorFitnessTest()
        {
            Net Net = new Net(new Random(), 2, 1);

            BackPropagation prop = new BackPropagation(2, 1);
            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1
            var dataSets = new[] { _1, _2, _3, _4 };

            prop.AddDataSet(_1);
            prop.AddDataSet(_2);
            prop.AddDataSet(_3);
            prop.AddDataSet(_4);

            var fitnessEvaluator = new DatasetEvaluator(dataSets);

            var fitnessScore1 = fitnessEvaluator.EvaluateNet(new[] { Net }.ToList(), Net, new TrainingStatus(0, 0, 0, new TimeSpan(), new TimeSpan()));

            Assert.IsTrue(fitnessScore1 < 300000, $"Pre trained fitness score is not correct: {fitnessScore1}");

            int epoch = 1000;
            var backPropResult = prop.Train(Net, epoch);

            Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);

            var fitnessScore2 = fitnessEvaluator.EvaluateNet(new[] { backPropResult.Net }.ToList(), backPropResult.Net, new TrainingStatus(0, 0, 0, new TimeSpan(), new TimeSpan()));

            Assert.IsTrue(fitnessScore2 > 100000000, $"Post trained fitness score is not correct: {fitnessScore2}.");
        }

        [TestMethod]
        public void DeterministicDatasetEvaluatorTest()
        {
            Random random = new Random();
            Net Net = new Net(new Random(), 2, 1);

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1
            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);

            foreach(var i in Enumerable.Range(0, 1000))
            {
                Net net = new Net(new Random(), 2, 1);

                var scores = Enumerable.Range(0, 1000).Select(index => fitnessEvaluator.EvaluateNet(null, net, null)).ToArray();
                Assert.AreEqual(1, scores.Distinct().Count());
            }

        }


        [TestMethod]
        public void LogicNetOR()
        {
            int iterations = 1000;
            int population = 100;

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1
            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);
            var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

            var topNet = trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations).First();

            double sse = 0;
            foreach (var dataSet in dataSets)
            {
                double result = topNet.Calculate(dataSet.Inputs).First();
                sse += Math.Pow(dataSet.Outputs.First() - result, 2.0);
            }

            Assert.IsTrue(sse < 0.2, $"LogicNetOR SSE after {iterations} iterations is '{sse}'");
        }

        [TestMethod]
        public void LogicNetAND()
        {
            int iterations = 1000;
            int population = 100;

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0 });    // 0 | 1 = 0
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0 });    // 1 | 0 = 0
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1

            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);
            var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

            var topNet = trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations).First();

            double sse = 0;
            foreach (var dataSet in dataSets)
            {
                double result = topNet.Calculate(dataSet.Inputs).First();
                sse += Math.Pow(dataSet.Outputs.First() - result, 2.0);
            }

            Assert.IsTrue(sse < 0.2, $"LogicNetAND SSE after {iterations} iterations is '{sse}'");
        }

        [TestMethod]
        public void LogicNetXOR()
        {
            int iterations = 1000;
            int population = 100;

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 0 });    // 1 | 1 = 0

            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);
            var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

            var topNet = trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations).First();

            double sse = 0;
            foreach (var dataSet in dataSets)
            {
                double result = topNet.Calculate(dataSet.Inputs).First();
                sse += Math.Pow(dataSet.Outputs.First() - result, 2.0);
            }

            Assert.IsTrue(sse < 0.2, $"LogicNetXOR SSE after {iterations} iterations is '{sse}'");
        }

        [TestMethod]
        public void LogicNetANDORXOR()
        {
            int iterations = 1000;
            int population = 100;

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0, 0, 0 });    // 0 | 0 = 000
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 0, 1, 1 });    // 0 | 1 = 011
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 0, 1, 1 });    // 1 | 0 = 011
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1, 1, 0 });    // 1 | 1 = 110

            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);
            var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

            var topNet = trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations).First();

            double sse = 0;
            foreach (var dataSet in dataSets)
            {
                double result = topNet.Calculate(dataSet.Inputs).First();
                sse += Math.Pow(dataSet.Outputs.First() - result, 2.0);
            }

            Assert.IsTrue(sse < 0.2, $"LogicNetANDORXOR SSE after {iterations} iterations is '{sse}'");
        }

        /// <summary>
        /// Tests if nets unique after spawning.
        /// </summary>
        [TestMethod]
        public void SpawnedNetsAreUniqueTest()
        {
            int iterations = 1000;
            int population = 100;

            DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
            DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
            DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
            DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 1 });    // 1 | 1 = 1
            var dataSets = new[] { _1, _2, _3, _4 };

            var fitnessEvaluator = new DatasetEvaluator(dataSets);
            var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

            int iteration = 0;
            trainer.NetsSpawned += (sender, netsSpawnedEventArgs) =>
            {
                var spawnedNetCount = netsSpawnedEventArgs.Nets.Select(n => n.CreateSerialisedNet()).Distinct().Count();

                Assert.AreEqual(population, spawnedNetCount, $"Iteration: {iteration}");
            };

            trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations);
        }

    }
}
