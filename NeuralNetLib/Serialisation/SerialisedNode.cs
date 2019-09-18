using RichTea.Common;
using System.Linq;

namespace RichTea.NeuralNetLib.Serialisation
{
    /// <summary>
    /// A mutable node designed for serialising neural nodes.
    /// </summary>
    public class SerialisedNode
    {
        /// <summary>
        /// Gets or sets bias.
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public double[] Weights { get; set; }

        /// <summary>
        /// Creates a node from this serialised node.
        /// </summary>
        /// <returns></returns>
        public Node CreateNode()
        {
            var node = new SigmoidNode(Bias, Weights);
            return node;
        }

        /// <summary>
        /// Returns a string the represents the serialised node.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNode>(this)
                .Append(p => p.Bias)
                .Append(p => p.Weights)
                .ToString();
        }

        /// <summary>
        /// Determines if this object equals another.
        /// </summary>
        /// <param name="that">Object to compare to.</param>
        /// <returns></returns>
        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNode>(this, that)
                .Append(p => p.Bias)
                .Append(p => p.Weights)
                .AreEqual;
        }

        /// <summary>
        /// Gets hash code.
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            var hash = new HashCodeBuilder()
                .Append(Bias)
                .Append(Weights)
                .HashCode;
            return hash;
        }

        /// <summary>
        /// Determines if the objects are equal.
        /// </summary>
        /// <param name="lhs">First object to compare.</param>
        /// <param name="rhs">Second object to compare.</param>
        /// <returns></returns>
        public static bool operator ==(SerialisedNode lhs, SerialisedNode rhs)
        {
            if (ReferenceEquals(lhs, null))
            {
                return ReferenceEquals(rhs, null);
            }

            return lhs.Equals(rhs);
        }

        /// <summary>
        /// Determines if the objects are not equal.
        /// </summary>
        /// <param name="lhs">First object to compare.</param>
        /// <param name="rhs">Second object to compare.</param>
        /// <returns></returns>
        public static bool operator !=(SerialisedNode lhs, SerialisedNode rhs)
        {
            return !(lhs == rhs);
        }
    }
}
