using NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class NodeLayer
    {
        #region Properties

        public Node[] Nodes { get; private set;  }
        public int Inputs { get; private set; }
        public int Outputs
        {
            get { return Nodes.Length; }
        }

        #endregion

        #region Methods

        public NodeLayer(int inputs, int outputs)
        {
            if (inputs == 0)
            {
                throw new ArgumentException("A NodeLayer must have at least one input");
            }

            if (outputs == 0)
            {
                throw new ArgumentException("A NodeLayer must have at least one output");
            }

            Inputs = inputs;

            Nodes = new Node[outputs];
            for (int i = 0; i < outputs; i++)
            {
                Nodes[i] = new Node(inputs);
            }
        }

        public NodeLayer(IEnumerable<Node> nodes)
        {
            if (nodes.Select(n => n.Weights.Length).Distinct().Count() != 1)
            {
                throw new ArgumentException("Nodes in a node layer must have the same number of inputs.");
            }

            Inputs = nodes.Select(n => n.Weights.Length).First();

            Nodes = nodes.ToArray();
        }

        public void SeedWeights(Random random)
        {
            foreach(var node in Nodes)
            {
                node.SeedWeights(random);
            }
        }

        public SerialisedNodeLayer CreateSerialisedNodeLayer()
        {
            var serialisedNodes = Nodes.Select(n => n.CreateSerialisedNode()).ToArray();
            var serialisedNodeLayer = new SerialisedNodeLayer
            {
                Nodes = serialisedNodes
            };
            return serialisedNodeLayer;
        }

        public void SeedWeights(NodeLayer nodeLayer)
        {
            if (nodeLayer.Inputs != Inputs)
            {
                throw new ArgumentException("NodeLayer has incorrect number of inputs.");
            }
            if (nodeLayer.Outputs != Outputs)
            {
                throw new ArgumentException("NodeLayer has incorrect number of outputs.");
            }

            for (int i = 0; i < Inputs; i++)
            {
                Nodes[i].SeedWeights(nodeLayer.Nodes[i]);
            }
        }

        public double[] Calculate(double[] inputs)
        {
            if (Inputs != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            double[] results = new double[Outputs];
            for (int i = 0; i < Outputs; i++)
            {
                results[i] = Nodes[i].Calculate(inputs);
            }
            return results;
        }

        public double[] Calculate(double[] inputs, double[] targets, ref double sse)
        {
            if (Inputs != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            sse = 0;
            double[] results = new double[Outputs];
            for (int i = 0; i < Outputs; i++)
            {
                double error = 0;
                results[i] = Nodes[i].Calculate(inputs, targets[i], ref error);
                sse += Math.Pow(error, 2);
            }
            return results;
        }

        #endregion

    }
}
