using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public interface INodeLayer
    {
        #region Properties

        INode[] Nodes { get; }

        int Inputs { get; }
        int Outputs { get; }

        #endregion

        #region Methods

        double[] Calculate(double[] Inputs);

        #endregion
    }
}
