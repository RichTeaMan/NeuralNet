using RichTea.Common.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class NetFactory
    {

        public Net GenerateRandomNet(int inputCount, int outputCount, Random random)
        {
            Net net = new Net(inputCount, outputCount);
            net.SeedWeights(random);
            return net;
        }

        public List<Net> GenerateRandomNetList(int inputCount, int outputCount, Random random, int netCount)
        {
            List<Net> netList = new List<Net>();
            foreach(var i in Enumerable.Range(0, netCount))
            {
                Net net = GenerateRandomNet(inputCount, outputCount, random);
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
        public Net CreateMutatedNet(Net net, Random random, double deviation)
        {
            Net mutatedNet = new Net(net.Inputs, net.Outputs, net.Layers);
            mutatedNet.SeedWeights(net);

            foreach(var nodeLayer in mutatedNet.NodeLayers)
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
            return mutatedNet;
        }

    }
}
