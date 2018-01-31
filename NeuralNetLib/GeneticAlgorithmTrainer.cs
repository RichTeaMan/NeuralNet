using RichTea.NeuralNetLib.Mutators;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    public class GeneticAlgorithmTrainer
    {
        public int threads { get; set; } = Environment.ProcessorCount;
        public int? seed = null;

        public List<Net> TrainAi(
            int inputCount,
            int outputCount,
            int hiddenLayers,
            int generationCount,
            int iterationCount,
            IFitnessEvaluator fitnessEvaluator
            )
        {
            Random random;
            if (seed.HasValue)
            {
                Console.WriteLine($"Random seed: {seed}");
                random = new Random(seed.Value);
            }
            else
            {
                random = new Random();
            }

            RandomMutator randomMutator = new RandomMutator(random)
            {
                Deviation = 0.00001
            };
            SplitChromosomeMutator splitChromosomeMutator = new SplitChromosomeMutator(random);

            List<Net> contestants = new NetFactory().GenerateRandomNetList(inputCount, outputCount, hiddenLayers, random, generationCount);

            Stopwatch trainerStopwatch = new Stopwatch();
            trainerStopwatch.Start();

            int totalEvaluations = 0;
            foreach (var currentIteration in Enumerable.Range(0, iterationCount))
            {
                // TODO: Add event
                Console.WriteLine($"Iteration {currentIteration} underway.");

                int netsProcessedInGeneration = 0;

                ConcurrentBag<EvaluatedNet> netEvaluations = new ConcurrentBag<EvaluatedNet>();

                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                ParallelOptions parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = threads
                };
                Parallel.ForEach(contestants, parallelOptions, (contestant, loopState) =>
                {
                    var trainingStatus = new TrainingStatus(currentIteration, netsProcessedInGeneration, totalEvaluations, stopwatch.Elapsed, trainerStopwatch.Elapsed);

                    var competingContestants = contestants.Where(n => !ReferenceEquals(n, contestant)).ToList().AsReadOnly();
                    int fitnessScore = fitnessEvaluator.EvaluateNet(competingContestants, contestant, trainingStatus);

                    var evaluatedNet = new EvaluatedNet(contestant, fitnessScore);
                    netEvaluations.Add(evaluatedNet);

                    Interlocked.Increment(ref netsProcessedInGeneration);
                    Interlocked.Increment(ref totalEvaluations);
                });
                stopwatch.Stop();

                var orderedContestants = netEvaluations.OrderByDescending(ne => ne.FitnessScore).ToList();

                // TODO: Add event
                Console.WriteLine();
                Console.WriteLine("Generation complete.");

                // write nets?

                var nextContestants = new List<Net>();
                foreach (var contestantI in orderedContestants.Take(generationCount / 2))
                {
                    Net spawnedNet;
                    if (currentIteration % 2 == 0)
                    {
                        spawnedNet = randomMutator.GenetateMutatedNeuralNet(contestantI.Net);
                    }
                    else
                    {
                        // get random second parent
                        int pick = random.Next(generationCount / 2);
                        var secondNet = orderedContestants[pick].Net;
                        spawnedNet = splitChromosomeMutator.GenetateMutatedNeuralNet(contestantI.Net, secondNet);
                    }

                    nextContestants.Add(spawnedNet);
                }
                contestants = nextContestants;
            }
            trainerStopwatch.Stop();

            // TODO add event

            Console.WriteLine($"Training complete.");
            return contestants;
        }

    }
}
