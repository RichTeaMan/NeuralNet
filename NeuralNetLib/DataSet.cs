using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class DataSet
    {
        public double[] Inputs { get; private set; }

        public int InputCount
        {
            get { return Inputs.Length; }
        }

        public double[] Outputs { get; private set; }

        public int OutputCount
        {
            get { return Outputs.Length; }
        }

        public DataSet(double[] Inputs, double[] Outputs)
        {
            this.Inputs = Inputs;
            this.Outputs = Outputs;
        }
    }
}
