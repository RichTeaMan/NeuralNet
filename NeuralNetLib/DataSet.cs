using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// Represents a colliection of data iwht inputs and expected outputs.
    /// </summary>
    public class DataSet
    {
        private List<double> _inputs;
        /// <summary>
        /// Gets inputs.
        /// </summary>
        public double[] Inputs { get{return _inputs.ToArray();}}

        /// <summary>
        /// Gets input count.
        /// </summary>
        public int InputCount
        {
            get { return _inputs.Count(); }
        }

        private Dictionary<string, int> _inputNames;

        /// <summary>
        /// Sets name of an input in the data set.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="name"></param>
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

        /// <summary>
        /// Finds an index is within the input count.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public bool IsIndexInInputRange(int index)
        {
            return !(index < 0 || index >= InputCount);
        }

        /// <summary>
        /// Finds input bu name.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public double GetInputByName(string name)
        {
            return _inputs[_inputNames[name]];
        }

        /// <summary>
        /// Sets input by a name.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        public void SetInputByName(string name, double value)
        {
            _inputs[_inputNames[name]] = value;
        }

        /// <summary>
        /// Deletes an input by an index.
        /// </summary>
        /// <param name="index"></param>
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

        /// <summary>
        /// Deletes an input by a name.
        /// </summary>
        /// <param name="name"></param>
        public void DeleteInputByName(string name)
        {
            if(!_inputNames.ContainsKey(name))
            {
                throw new ArgumentException("No input with that name exists.");
            }
            int index = _inputNames[name];
            DeleteInputByIndex(index);
        }

        /// <summary>
        /// Adds an input.
        /// </summary>
        /// <param name="value"></param>
        public void AddInput(double value)
        {
            _inputs.Add(value);
        }

        /// <summary>
        /// Adds an input with a name.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        public void AddInput(string name, double value)
        {
            int index = InputCount;
            AddInput(value);
            SetName(index, name);
        }

        private List<double> _outputs;
        /// <summary>
        /// Gets outputs.
        /// </summary>
        public double[] Outputs { get{return _outputs.ToArray();} }

        /// <summary>
        /// Gets output count.
        /// </summary>
        public int OutputCount
        {
            get { return _outputs.Count(); }
        }

        /// <summary>
        /// Initialises a dataset.
        /// </summary>
        public DataSet()
        {
            _inputs = new List<double>();
            _outputs = new List<double>();
            _inputNames = new Dictionary<string, int>();
        }

        /// <summary>
        /// Initialises dataset from data.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        public DataSet(double[] inputs, double[] outputs)
        {
            _inputs = new List<double>(inputs);
            _outputs = new List<double>(outputs);
            _inputNames = new Dictionary<string, int>();
        }
    }
}
