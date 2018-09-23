using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// Single layer collection of <see cref="Node" />s.
    /// </summary>
    public class NodeLayer
    {
        #region Properties

        /// <summary>
        /// Gets nodes in the layer.
        /// </summary>
        public IReadOnlyList<Node> Nodes { get; }

        /// <summary>
        /// Gets input count.
        /// </summary>
        public int InputCount { get; }

        /// <summary>
        /// Gets output count.
        /// </summary>
        public int OutputCount
        {
            get { return Nodes.Count; }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Initialises node layer.
        /// </summary>
        /// <param name="inputCount">Input count.</param>
        /// <param name="outputCount">Output count.</param>
        /// <param name="random">Random.</param>
        public NodeLayer(int inputCount, int outputCount, Random random)
        {
            if (inputCount <= 0)
            {
                throw new ArgumentException("A NodeLayer must have at least one input");
            }

            if (outputCount <= 0)
            {
                throw new ArgumentException("A NodeLayer must have at least one output");
            }

            InputCount = inputCount;

            Nodes = Enumerable.Range(0, outputCount).Select(i => new Node(inputCount, random)).ToList();
        }

        /// <summary>
        /// Initialises node layer.
        /// </summary>
        /// <param name="nodes">Nodes.</param>
        public NodeLayer(IEnumerable<Node> nodes)
        {
            if (nodes.Select(n => n.Weights.Count).Distinct().Count() != 1)
            {
                throw new ArgumentException("Nodes in a node layer must have the same number of inputs.");
            }

            InputCount = nodes.Select(n => n.Weights.Count).First();

            Nodes = nodes.ToArray();
        }

        /// <summary>
        /// Creates node layer for serialisation.
        /// </summary>
        /// <returns></returns>
        public SerialisedNodeLayer CreateSerialisedNodeLayer()
        {
            var serialisedNodes = Nodes.Select(n => n.CreateSerialisedNode()).ToArray();
            var serialisedNodeLayer = new SerialisedNodeLayer
            {
                Nodes = serialisedNodes
            };
            return serialisedNodeLayer;
        }

        /// <summary>
        /// Calculates result from inputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <returns>Result.</returns>
        public double[] Calculate(double[] inputs)
        {
            if (InputCount != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            double[] results = new double[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                results[i] = Nodes[i].Calculate(inputs);
            }
            return results;
        }

        /// <summary>
        /// Calculates result from inputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="targets">Target.</param>
        /// <param name="sse">SEE, calculated from the result and the target.</param>
        /// <returns>Result.</returns>
        public double[] Calculate(double[] inputs, double[] targets, ref double sse)
        {
            if (InputCount != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            sse = 0;
            double[] results = new double[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                double error = 0;
                results[i] = Nodes[i].Calculate(inputs, targets[i], ref error);
                sse += Math.Pow(error, 2);
            }
            return results;
        }

        #endregion

        /// <summary>
        /// Generates string representation from node layer.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<NodeLayer>(this)
                .Append(p => p.Nodes)
                .ToString();
        }

    }
}
