using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public interface INode
    {
        #region Properties

        int Inputs { get; }
        double Bias { get; set;  }
        double[] Weights { get; set; }
        double Result { get; }

        #endregion

        #region Methods

        double Calculate(double[] Inputs);
        double Calculate(double[] Inputs, double Target, ref double Error);

        #endregion
    }
}
