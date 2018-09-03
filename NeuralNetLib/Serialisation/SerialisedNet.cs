using RichTea.Common;
using System.Linq;

namespace RichTea.NeuralNetLib.Serialisation
{
    /// <summary>
    /// A mutable net designed for serialising neural nets.
    /// </summary>
    public class SerialisedNet
    {
        /// <summary>
        /// Gets or sets node layers.
        /// </summary>
        public SerialisedNodeLayer[] NodeLayers { get; set; }

        /// <summary>
        /// Creates a net from this serialised net.
        /// </summary>
        /// <returns></returns>
        public Net CreateNet()
        {
            var nodeLayers = NodeLayers.Select(n => n.CreateNodeLayer());
            var net = new Net(nodeLayers);
            return net;
        }

        /// <summary>
        /// Returns a string the represents the serialised net.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNet>(this)
                .Append(p => p.NodeLayers)
                .ToString();
        }

        /// <summary>
        /// Determines if this object equals another.
        /// </summary>
        /// <param name="that">Object to compare to.</param>
        /// <returns></returns>
        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNet>(this, that)
                .Append(p => p.NodeLayers)
                .AreEqual;
        }

        /// <summary>
        /// Gets hash code.
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            var hashCodeBuilder = new HashCodeBuilder();

            var weights = NodeLayers.SelectMany(nl => nl.Nodes.SelectMany(n => n.Weights).ToArray()).ToArray();
            var biases = NodeLayers.SelectMany(nl => nl.Nodes.Select(n => n.Bias)).ToArray();

            hashCodeBuilder.Append(weights).Append(biases);
            return hashCodeBuilder.HashCode;
        }

        /// <summary>
        /// Determines if two objects are equal.
        /// </summary>
        /// <param name="lhs">First object to compare.</param>
        /// <param name="rhs">Second object to compare.</param>
        /// <returns></returns>
        public static bool operator ==(SerialisedNet lhs, SerialisedNet rhs)
        {
            if (ReferenceEquals(lhs, null))
            {
                return ReferenceEquals(rhs, null);
            }

            return lhs.Equals(rhs);
        }

        /// <summary>
        /// Determins if the two objects are not equal.
        /// </summary>
        /// <param name="lhs">First object to compare.</param>
        /// <param name="rhs">Second object to compare.</param>
        /// <returns></returns>
        public static bool operator !=(SerialisedNet lhs, SerialisedNet rhs)
        {
            return !(lhs == rhs);
        }
    }
}
