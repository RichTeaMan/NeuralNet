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

        #endregion

        #region Methods

        double Calculate(double[] Inputs);

        #endregion
    }
}
