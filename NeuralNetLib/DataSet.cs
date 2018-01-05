using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    public class DataSet
    {
        private List<double> _inputs;
        public double[] Inputs { get{return _inputs.ToArray();}}

        public int InputCount
        {
            get { return _inputs.Count(); }
        }

        private Dictionary<string, int> _inputNames;

        public void SetName(int index, string name)
        {
            if(!IsIndexInInputRange(index))
            {
                throw new ArgumentOutOfRangeException("index is not within range of inputs.");
            }
            if(!_inputNames.ContainsKey(name))
            {
                throw new ArgumentException("A each name must be unique.");
            }
            _inputNames.Add(name, index);
        }

        public bool IsIndexInInputRange(int index)
        {
            return !(index < 0 || index >= InputCount);
        }

        public double GetInputByName(string name)
        {
            return _inputs[_inputNames[name]];
        }

        public void SetInputByName(string name, double value)
        {
            _inputs[_inputNames[name]] = value;
        }

        public void DeleteInputByIndex(int index)
        {
            if(!IsIndexInInputRange(index))
            {
                throw new ArgumentOutOfRangeException("index is not within range of inputs.");
            }
            _inputs.RemoveAt(index);

            // delete name if necessary
            string name  = _inputNames.Where(i => i.Value == index).Select(i => i.Key).FirstOrDefault();
            if(name != null)
                _inputNames.Remove(name);
        }

        public void DeleteInputByName(string name)
        {
            if(!_inputNames.ContainsKey(name))
            {
                throw new ArgumentException("No input with that name exists.");
            }
            int index = _inputNames[name];
            DeleteInputByIndex(index);
        }

        public void AddInput(double value)
        {
            _inputs.Add(value);
        }

        public void AddInput(string name, double value)
        {
            int index = InputCount;
            AddInput(value);
            SetName(index, name);
        }

        private List<double> _outputs;
        public double[] Outputs { get{return _outputs.ToArray();} }

        public int OutputCount
        {
            get { return _outputs.Count(); }
        }

        public DataSet()
        {
            _inputs = new List<double>();
            _outputs = new List<double>();
        }

        public DataSet(double[] Inputs, double[] Outputs)
        {
            _inputs = new List<double>(Inputs);
            _outputs = new List<double>(Outputs);
        }
    }
}
