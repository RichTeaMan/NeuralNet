using RichTea.Common.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Factory for generating nets.
    /// </summary>
    public class NetFactory
    {

        /// <summary>
        /// Generates a random net.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="random">Random.</param>
        /// <returns>Net.</returns>
        public Net GenerateRandomNet(int inputCount, int outputCount, Random random)
        {
            Net net = new Net(random, inputCount, outputCount);
            return net;
        }

        /// <summary>
        /// Generates a random net.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="hiddenLayers">Hidden layers.</param>
        /// <param name="random">Random.</param>
        /// <returns>Net.</returns>
        public Net GenerateRandomNet(int inputCount, int outputCount, int hiddenLayers, Random random)
        {
            Net net = new Net(random, inputCount, outputCount, hiddenLayers + 2);
            return net;
        }

        /// <summary>
        /// Generates a list of random net.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="random">Random.</param>
        /// <param name="netCount">Number of nets to generate.</param>
        /// <returns>List of nets.</returns>
        public List<Net> GenerateRandomNetList(int inputCount, int outputCount, Random random, int netCount)
        {
            List<Net> netList = GenerateRandomNetList(inputCount, outputCount, 1, random, netCount);
            return netList;
        }

        /// <summary>
        /// Generates a list of random net.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="hiddenLayers">Hidden layers.</param>
        /// <param name="random">Random.</param>
        /// <param name="netCount">Number of nets to generate.</param>
        /// <returns>List of nets.</returns>
        public List<Net> GenerateRandomNetList(int inputCount, int outputCount, int hiddenLayers, Random random, int netCount)
        {
            List<Net> netList = new List<Net>();
            foreach(var i in Enumerable.Range(0, netCount))
            {
                Net net = GenerateRandomNet(inputCount, outputCount, hiddenLayers, random);
                netList.Add(net);
            }
            return netList;
        }

        /// <summary>
        /// Creates a new net from the given net, randomly changing the weights.
        /// 
        /// The supplied net does not get modified.
        /// </summary>
        /// <param name="net">Net to copy and mutate.</param>
        /// <param name="random">Random.</param>
        /// <param name="deviation">Deviation to muate by.</param>
        /// <returns></returns>
        [Obsolete("Use RandomMutator")]
        public Net CreateMutatedNet(Net net, Random random, double deviation)
        {
            var serialisedNet = net.CreateSerialisedNet();

            foreach(var nodeLayer in serialisedNet.NodeLayers)
            {
                foreach(var node in nodeLayer.Nodes)
                {
                    node.Bias += random.NextDoubleInRange(deviation);
                    for(int i = 0; i < node.Weights.Length; i++)
                    {
                        node.Weights[i] += random.NextDoubleInRange(deviation);
                    }
                }
            }
            var mutatedNet = serialisedNet.CreateNet();
            return mutatedNet;
        }

    }
}
