using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public interface INet
    {
        #region Properties

        INodeLayer[] NodeLayers { get; }

        int Inputs { get; }
        int Outputs { get; }
        int Layers { get; }

        #endregion

        #region Methods

        double[] Calculate(double[] Inputs);

        #endregion
    }
}
