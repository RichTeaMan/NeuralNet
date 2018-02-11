using RichTea.NeuralNetLib.Mutators;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Generalised genetic algorithm trainer for neural nets.
    /// </summary>
    /// <typeparam name="T">Fitness evaluator for sorting nets.</typeparam>
    public class GeneticAlgorithmTrainer<T> where T : IFitnessEvaluator
    {
        /// <summary>
        /// Gets or sets the amount of threads to use. Defaults to prcoessor thread count.
        /// </summary>
        public int Threads { get; set; } = Environment.ProcessorCount;

        /// <summary>
        /// Gets or sets the seed for the random number generator.
        /// </summary>
        public int? Seed { get; set; } = null;

        /// <summary>
        /// Gets the fitness evaluator used in the trainer.
        /// </summary>
        public T FitnessEvaluator { get; }

        /// <summary>
        /// Gets a readonly list of mutators used in the trainer.
        /// </summary>
        public IReadOnlyList<INeuralNetMutator> Mutators { get; }

        /// <summary>
        /// Represents the method that will handle iteration started events.
        /// </summary>
        /// <param name="sender">The trainer that invoked the event.</param>
        /// <param name="iterationStartedEventArgs">Event arguments.</param>
        public delegate void IterationStartedEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationStartedEventArgs iterationStartedEventArgs);

        /// <summary>
        /// Occurs when the iteration has started.
        /// </summary>
        public event IterationStartedEventArgsHandler IterationStarted;

        /// <summary>
        /// Represents the method that will handle iteration in progress events.
        /// </summary>
        /// <param name="sender">The trainer that invoked the event.</param>
        /// <param name="iterationInProgressEventArgs">Event arguments.</param>
        public delegate void IterationInProgressEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationInProgressEventArgs iterationInProgressEventArgs);

        /// <summary>
        /// Occurs regularly when the iteration is in progress. This event will be fired very frequently, for performance reasons and consumers of this should exit as fast as possible.
        /// </summary>
        public event IterationInProgressEventArgsHandler IterationInProgress;

        /// <summary>
        /// Represents the method that will handle iteration complete events.
        /// </summary>
        /// <param name="sender">The trainer that invoked the event.</param>
        /// <param name="iterationCompleteEventArgs">Event arguments.</param>
        public delegate void IterationCompleteEventArgsHandler(GeneticAlgorithmTrainer<T> sender, IterationCompleteEventArgs iterationCompleteEventArgs);

        /// <summary>
        /// Occurs when the iteration is complete.
        /// </summary>
        public event IterationCompleteEventArgsHandler IterationComplete;

        /// <summary>
        /// Represents the method that will handle nets spawned events.
        /// </summary>
        /// <param name="sender">The trainer that invoked the event.</param>
        /// <param name="netsSpawnedEventArgs">Event arguments.</param>
        public delegate void NetsSpawnedEventArgsHandler(GeneticAlgorithmTrainer<T> sender, NetsSpawnedEventArgs netsSpawnedEventArgs);

        /// <summary>
        /// Occurs when nets have been spawned.
        /// </summary>
        public event NetsSpawnedEventArgsHandler NetsSpawned;

        /// <summary>
        /// Random.
        /// </summary>
        private Random _random;

        public GeneticAlgorithmTrainer(T fitnessEvaluator)
        {
            FitnessEvaluator = fitnessEvaluator;
            if (Seed.HasValue)
            {
                Console.WriteLine($"Random seed: {Seed}");
                _random = new Random(Seed.Value);
            }
            else
            {
                _random = new Random();
            }


            var randomMutator = new RandomMutator(_random)
            {
                Deviation = 0.00001
            };
            var splitChromosomeMutator = new SplitChromosomeMutator(_random);
            var singularRandomNodeMutator = new SingularRandomNodeMutator(_random);
            var weakestNodeMutator = new WeakestNodeMutator(_random);
            var crossoverNodesMutator = new CrossoverNodesMutator(_random);

            Mutators = new List<INeuralNetMutator>() {
                randomMutator,
                splitChromosomeMutator,
                singularRandomNodeMutator,
                weakestNodeMutator,
                crossoverNodesMutator
            };
        }

        public List<Net> TrainAi(
            int inputCount,
            int outputCount,
            int hiddenLayers,
            int generationCount,
            int iterationCount
            )
        {

            List<Net> contestants = new NetFactory().GenerateRandomNetList(inputCount, outputCount, hiddenLayers, _random, generationCount);
            NetsSpawned?.Invoke(this, new NetsSpawnedEventArgs(contestants, null));

            Stopwatch trainerStopwatch = new Stopwatch();
            trainerStopwatch.Start();

            int totalEvaluations = 0;

            var mutatorEnumerator = Mutators.GetEnumerator();
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

                if (!mutatorEnumerator.MoveNext())
                {
                    mutatorEnumerator.Reset();
                    mutatorEnumerator.MoveNext();
                }
                var mutator = mutatorEnumerator.Current;
                var nextContestants = new List<Net>();
                foreach (var contestantI in orderedContestants.Take(generationCount / 2))
                {
                    Net spawnedNet;

                    if (mutator is INeuralNetOneParentMutator oneParentMutator)
                    {
                        spawnedNet = oneParentMutator.GenetateMutatedNeuralNet(contestantI.Net);
                    }
                    else if (mutator is INeuralNetTwoParentMutator twoParentMutator)
                    {
                        // get random second parent
                        int pick = _random.Next(generationCount / 2);
                        var secondNet = orderedContestants[pick].Net;
                        spawnedNet = twoParentMutator.GenetateMutatedNeuralNet(contestantI.Net, secondNet);
                    }
                    else
                    {
                        throw new Exception("Unknown mutator interface.");
                    }
                    nextContestants.Add(contestantI.Net);
                    nextContestants.Add(spawnedNet);
                }
                contestants = nextContestants;
                NetsSpawned?.Invoke(this, new NetsSpawnedEventArgs(contestants, mutator));

            }
            trainerStopwatch.Stop();

            return contestants;
        }

    }
}
