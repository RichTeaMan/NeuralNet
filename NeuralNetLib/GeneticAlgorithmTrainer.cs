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
    public class GeneticAlgorithmTrainer<T> where T : IFitnessEvaluator
    {
        public int Threads { get; set; } = Environment.ProcessorCount;
        public int? Seed { get; set; } = null;

        public T FitnessEvaluator { get; }

        public delegate void IterationStartedEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationStartedEventArgs iterationStartedEventArgs);

        public event IterationStartedEventArgsHandler IterationStarted;

        public delegate void IterationInProgressEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationInProgressEventArgs iterationInProgressEventArgs);

        public event IterationInProgressEventArgsHandler IterationInProgress;

        public delegate void IterationCompleteEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationCompleteEventArgs iterationCompleteEventArgs);

        public event IterationCompleteEventArgsHandler IterationComplete;

        public GeneticAlgorithmTrainer(T fitnessEvaluator)
        {
            FitnessEvaluator = fitnessEvaluator;
        }

        public List<Net> TrainAi(
            int inputCount,
            int outputCount,
            int hiddenLayers,
            int generationCount,
            int iterationCount
            )
        {
            Random random;
            if (Seed.HasValue)
            {
                Console.WriteLine($"Random seed: {Seed}");
                random = new Random(Seed.Value);
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
                int netsProcessedInGeneration = 0;

                IterationStarted?.Invoke(this, new IterationStartedEventArgs(new TrainingStatus(currentIteration, netsProcessedInGeneration, totalEvaluations, new TimeSpan(), trainerStopwatch.Elapsed)));

                ConcurrentBag<EvaluatedNet> netEvaluations = new ConcurrentBag<EvaluatedNet>();

                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                ParallelOptions parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = Threads
                };
                Parallel.ForEach(contestants, parallelOptions, (contestant, loopState) =>
                {
                    var trainingStatus = new TrainingStatus(currentIteration, netsProcessedInGeneration, totalEvaluations, stopwatch.Elapsed, trainerStopwatch.Elapsed);

                    var competingContestants = contestants.Where(n => !ReferenceEquals(n, contestant)).ToList().AsReadOnly();
                    int fitnessScore = FitnessEvaluator.EvaluateNet(competingContestants, contestant, trainingStatus);

                    var postEvaltrainingStatus = new TrainingStatus(currentIteration, netsProcessedInGeneration, totalEvaluations, stopwatch.Elapsed, trainerStopwatch.Elapsed);

                    IterationInProgress?.Invoke(this, new IterationInProgressEventArgs(postEvaltrainingStatus));

                    var evaluatedNet = new EvaluatedNet(contestant, fitnessScore);
                    netEvaluations.Add(evaluatedNet);

                    Interlocked.Increment(ref netsProcessedInGeneration);
                    Interlocked.Increment(ref totalEvaluations);
                });
                stopwatch.Stop();

                var orderedContestants = netEvaluations.OrderByDescending(ne => ne.FitnessScore).ToList();

                IterationComplete?.Invoke(this, new IterationCompleteEventArgs(orderedContestants, new TrainingStatus(currentIteration, netsProcessedInGeneration, totalEvaluations, stopwatch.Elapsed, trainerStopwatch.Elapsed)));

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

            return contestants;
        }

    }
}
