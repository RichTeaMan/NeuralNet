using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Neural net.
    /// </summary>
    public class Net
    {
        #region Properties

        /// <summary>
        /// Gets or sets if output should be normalised.
        /// If true, all output is guaranteed to be between 0 and 1.0 but output may not be deterministic until the net
        /// has seen all extremes of data.
        /// 
        /// Defaults to false.
        /// </summary>
        public bool NormaliseOutput { get; set; } = false;

        /// <summary>
        /// Gets node layers.
        /// </summary>
        public NodeLayer[] NodeLayers { get; }

        /// <summary>
        /// Gets input counts.
        /// </summary>
        public int InputCount { get; }

        /// <summary>
        /// Gets output counts.
        /// </summary>
        public int OutputCount { get; }

        /// <summary>
        /// Gets layer count.
        /// </summary>
        public int Layers
        {
            get { return NodeLayers.Length; }
        }

        /// <summary>
        /// Gets node count.
        /// </summary>
        public int NodeCount
        {
            get
            {
                return NodeLayers.Sum(nl => nl.Nodes.Length);
            }
        }

        /// <summary>
        /// Gets all nodes in the net.
        /// </summary>
        public IEnumerable<Node> Nodes
        {
            get
            {
                foreach (var nodeLayer in NodeLayers)
                {
                    foreach (var node in nodeLayer.Nodes)
                    {
                        yield return node;
                    }
                }
            }
        }

        private NormaliserNode[] normaliserNodes;

        #endregion

        #region Methods

        /// <summary>
        /// Initialises net.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="layerCount">Layer count.</param>
        public Net(int inputCount, int outputCount, int layerCount = 3)
        {
            if (layerCount < 2)
            {
                throw new ArgumentException("There must be at least 2 layers.");
            }

            InputCount = inputCount;
            OutputCount = outputCount;

            NodeLayers = new NodeLayer[layerCount];
            for (int i = 0; i < layerCount - 1; i++)
            {
                NodeLayers[i] = new NodeLayer(inputCount, inputCount);
            }
            NodeLayers[layerCount - 1] = new NodeLayer(inputCount, outputCount);

            normaliserNodes = Enumerable.Range(0, InputCount).Select(i => new NormaliserNode()).ToArray();
        }

        /// <summary>
        /// Initialises net.
        /// </summary>
        /// <param name="nodeLayers">Node layers.</param>
        public Net(IEnumerable<NodeLayer> nodeLayers)
        {
            if (nodeLayers.Count() < 2)
            {
                throw new ArgumentException("There must be at least 2 layers.");
            }

            InputCount = nodeLayers.First().InputCount;
            OutputCount = nodeLayers.Last().OutputCount;

            NodeLayers = nodeLayers.ToArray();

            normaliserNodes = Enumerable.Range(0, InputCount).Select(i => new NormaliserNode()).ToArray();
        }

        /// <summary>
        /// Seeds weights randomly.
        /// </summary>
        /// <param name="random">Random.</param>
        public void SeedWeights(Random random)
        {
            foreach (var nodeLayer in NodeLayers)
            {
                nodeLayer.SeedWeights(random);
            }
        }

        /// <summary>
        /// Created a net for serialisation.
        /// </summary>
        /// <returns></returns>
        public SerialisedNet CreateSerialisedNet()
        {
            var serialisedNodeLayers = NodeLayers.Select(n => n.CreateSerialisedNodeLayer()).ToArray();
            var serialisedNet = new SerialisedNet
            {
                NodeLayers = serialisedNodeLayers
            };
            return serialisedNet;
        }

        /// <summary>
        /// Seed weights from another net.
        /// </summary>
        /// <param name="net">Net.</param>
        public void SeedWeights(Net net)
        {
            if (net.InputCount != InputCount)
            {
                throw new ArgumentException("Net has incorrect number of inputs.");
            }
            if (net.OutputCount != OutputCount)
            {
                throw new ArgumentException("Net has incorrect number of outputs.");
            }

            for (int i = 0; i < NodeLayers.Length; i++)
            {
                NodeLayers[i].SeedWeights(net.NodeLayers[i]);
            }
        }

        /// <summary>
        /// Calculate result from inputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <returns>Result.</returns>
        public double[] Calculate(double[] inputs)
        {
            if (InputCount != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            double[] interStep = inputs.ToArray();
            if (NormaliseOutput)
            {
                foreach (var i in Enumerable.Range(0, InputCount))
                {
                    interStep[i] = normaliserNodes[i].Calculate(inputs[i]);
                }
            }

            foreach (var layer in NodeLayers)
            {
                interStep = layer.Calculate(interStep);
            }
            return interStep;
        }

        /// <summary>
        /// Calculates from inputs. SSE will be calculated from targets.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="targets">Targets.</param>
        /// <param name="sse">SSE.</param>
        /// <returns>Resultrs.</returns>
        public double[] Calculate(double[] inputs, double[] targets, ref double sse)
        {
            if (targets.Length != OutputCount)
            {
                throw new ArgumentException("There is an incorrect number of Targets.");
            }

            var results = Calculate(inputs);

            sse = 0;
            for (int i = 0; i < results.Length; i++)
            {
                double error = targets[i] - results[i];
                sse += Math.Pow(error, 2);
            }

            return results;
        }

        /// <summary>
        /// Calculates if a net is equivalent.
        /// </summary>
        /// <param name="net">Net.</param>
        /// <returns>Is equivalent.</returns>
        public bool IsEquivalent(Net net)
        {
            var serialNet = CreateSerialisedNet();
            var otherSerialNet = net?.CreateSerialisedNet();

            return serialNet.Equals(otherSerialNet);
        }

        #endregion

        /// <summary>
        /// Converts net to a string.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<Net>(this)
                .Append(p => p.NodeLayers)
                .ToString();
        }

    }
}
