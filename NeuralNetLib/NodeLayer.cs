using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class NodeLayer : INodeLayer
    {
        #region Properties

        public INode[] Nodes { get; private set;  }
        public int Inputs { get; private set; }
        public int Outputs
        {
            get { return Nodes.Length; }
        }

        #endregion

        #region Methods

        public NodeLayer(int Inputs, int Outputs)
        {
            this.Inputs = Inputs;
            
            Nodes = new Node[Outputs];
            for (int i = 0; i < Inputs; i++)
            {
                Nodes[i] = new Node(Inputs);
            }
        }

        public double[] Calculate(double[] Inputs)
        {
            if (this.Inputs != Inputs.Length)
                throw new ArgumentException("There is an incorrect number of Inputs.");

            double[] results = new double[Outputs];
            for (int i = 0; i < Outputs; i++)
            {
                results[i] = Nodes[i].Calculate(Inputs);
            }
            return results;
        }

        #endregion

    }
}
