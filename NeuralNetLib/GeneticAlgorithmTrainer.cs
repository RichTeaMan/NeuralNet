using RichTea.NeuralNetLib.Mutators;
using RichTea.NeuralNetLib.Serialisation;
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
        /// Gets or sets if nets should be normalised.
        /// </summary>
        public bool NormaliseNets { get; set; }

        /// <summary>
        /// Gets or sets the amount of threads to use. Defaults to prcoessor thread count.
        /// </summary>
        public int Threads { get; set; } = Environment.ProcessorCount;

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

        /// <summary>
        /// Creates a list of mutators that are used as defaults.
        /// </summary>
        /// <param name="_random">Random.</param>
        /// <returns>Mutators.</returns>
        public static List<INeuralNetMutator> CreateDefaultMutators(Random _random)
        {
            var randomMutator = new RandomMutator(_random)
            {
                Deviation = 0.001
            };
            var splitChromosomeMutator = new SplitChromosomeMutator(_random);
            var singularRandomNodeMutator = new SingularRandomNodeMutator(_random);
            var weakestNodeMutator = new WeakestNodeMutator(_random);
            var crossoverNodesMutator = new CrossoverNodesMutator(_random);
            var smallRandomParameterMutator = new RandomParameterMutator() { ParameterAmount = 4 };
            var bigRandomParameterMutator = new RandomParameterMutator() { Deviation = 0.1 };
            var reallyRandomParameterMutator = new RandomParameterMutator() { Deviation = 1.0 };

            var mutators = new List<INeuralNetMutator>() {
                randomMutator,
                splitChromosomeMutator,
                singularRandomNodeMutator,
                weakestNodeMutator,
                crossoverNodesMutator,
                smallRandomParameterMutator,
                bigRandomParameterMutator,
                reallyRandomParameterMutator
            };
            return mutators;
        }

        /// <summary>
        /// Constructs genetic algoritm trainer.
        /// </summary>
        /// <param name="random">The random number generator to use.</param>
        /// <param name="fitnessEvaluator">Fitness evaluator to use.</param>
        public GeneticAlgorithmTrainer(Random random, T fitnessEvaluator) : this(random, fitnessEvaluator, CreateDefaultMutators(random))
        {
        }

        /// <summary>
        /// Constructs genetic algoritm trainer.
        /// </summary>
        /// <param name="random">The random number generator to use.</param>
        /// <param name="fitnessEvaluator">Fitness evaluator to use.</param>
        /// <param name="mutators">Mutators to use.</param>
        public GeneticAlgorithmTrainer(Random random, T fitnessEvaluator, IEnumerable<INeuralNetMutator> mutators)
        {
            FitnessEvaluator = fitnessEvaluator;
            _random = random;
            Mutators = mutators.ToList();
        }

        /// <summary>
        /// Trains neural nets with the given parameters.
        /// </summary>
        /// <param name="inputCount">How many inputs a net should have.</param>
        /// <param name="outputCount">How many outputs a net should have.</param>
        /// <param name="hiddenLayers">How many hidden layers a net should have.</param>
        /// <param name="populationCount">How many nets will be tested in each iteration.</param>
        /// <param name="iterationCount">How many iterations the trainer should complete.</param>
        /// <returns></returns>
        public List<Net> TrainAi(
            int inputCount,
            int outputCount,
            int hiddenLayers,
            int populationCount,
            int iterationCount
        )
        {
            List<Net> contestants = new NetFactory().GenerateRandomNetList(inputCount, outputCount, hiddenLayers, _random, populationCount);
            return TrainAi(contestants, populationCount, iterationCount);
        }

        /// <summary>
        /// Trains neural nets with the given parameters.
        /// </summary>
        /// <param name="startingContestants">StartingContestants.</param>
        /// <param name="populationCount">How many nets will be tested in each iteration.</param>
        /// <param name="iterationCount">How many iterations the trainer should complete.</param>
        /// <returns></returns>
        public List<Net> TrainAi(
        IEnumerable<Net> startingContestants,
            int populationCount,
            int iterationCount
        )
        {
            var mutatorEnumerator = Mutators.GetEnumerator();
            List<Net> contestants = MutateContestants(populationCount, mutatorEnumerator, startingContestants.ToList());

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
                contestants = MutateContestants(populationCount, mutatorEnumerator, orderedContestants.Select(c => c.Net).ToList());

            }
            trainerStopwatch.Stop();

            return contestants;
        }

        private List<Net> MutateContestants(int populationCount, IEnumerator<INeuralNetMutator> mutatorEnumerator, List<Net> orderedContestants)
        {
            List<Net> contestants;
            if (!mutatorEnumerator.MoveNext())
            {
                mutatorEnumerator.Reset();
                mutatorEnumerator.MoveNext();
            }
            var mutator = mutatorEnumerator.Current;
            var topContestantList = orderedContestants.Take(populationCount / 10).ToList();
            var topContestantEnumerator = topContestantList.GetEnumerator();

            // create serialised nets as they override equality methods so uniqueness can be verified.
            var nextContestants = new HashSet<SerialisedNet>();
            while (topContestantEnumerator.MoveNext())
            {
                var serialNet = topContestantEnumerator.Current.CreateSerialisedNet();
                nextContestants.Add(serialNet);
            }
            topContestantEnumerator = topContestantList.GetEnumerator();
            int netCollisions = 0;
            while (nextContestants.Count < populationCount)
            {
                if (!topContestantEnumerator.MoveNext())
                {
                    topContestantEnumerator = topContestantList.GetEnumerator();
                    topContestantEnumerator.MoveNext();
                }
                var contestantI = topContestantEnumerator.Current;

                Net spawnedNet;

                if (mutator is INeuralNetOneParentMutator oneParentMutator)
                {
                    spawnedNet = oneParentMutator.GenetateMutatedNeuralNet(contestantI);
                }
                else if (mutator is INeuralNetTwoParentMutator twoParentMutator)
                {
                    // get random second parent
                    int pick = _random.Next(populationCount / 2);
                    var secondNet = orderedContestants[pick];
                    spawnedNet = twoParentMutator.GenetateMutatedNeuralNet(contestantI, secondNet);
                }
                else
                {
                    throw new Exception("Unknown mutator interface.");
                }

                var serialSpawnNet = spawnedNet.CreateSerialisedNet();
                if (!nextContestants.Contains(serialSpawnNet))
                {
                    nextContestants.Add(serialSpawnNet);
                }
                else
                {
                    netCollisions++;
                    if (netCollisions > 1000)
                    {
                        int netsToGenerate = populationCount - nextContestants.Count;
                        Console.WriteLine($"There were too many net collions. Generating nets {netsToGenerate} randomly...");
                        foreach(var i in Enumerable.Range(0, netsToGenerate))
                        {
                            var randomContestant = new Net(contestantI.InputCount, contestantI.OutputCount, contestantI.Layers);
                            randomContestant.SeedWeights(_random);
                            nextContestants.Add(randomContestant.CreateSerialisedNet());
                        }
                    }
                }
            }
            contestants = nextContestants.Select(n => n.CreateNet()).ToList();

            contestants.ForEach(c => c.NormaliseOutput = NormaliseNets);
            NetsSpawned?.Invoke(this, new NetsSpawnedEventArgs(contestants, mutator));
            return contestants;
        }
    }
}
