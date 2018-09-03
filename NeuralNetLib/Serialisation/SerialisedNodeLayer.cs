using RichTea.Common;
using System.Linq;

namespace RichTea.NeuralNetLib.Serialisation
{
    /// <summary>
    /// A mutable node layer designed for serialising neural node layers.
    /// </summary>
    public class SerialisedNodeLayer
    {
        /// <summary>
        /// Gets or sets nodes.
        /// </summary>
        public SerialisedNode[] Nodes { get; set; }

        /// <summary>
        /// Creates a ndoe layer from this serialised layer.
        /// </summary>
        /// <returns></returns>
        public NodeLayer CreateNodeLayer()
        {
            var nodes = Nodes.Select(n => n.CreateNode()).ToArray();

            var nodeLayer = new NodeLayer(nodes);
            return nodeLayer;
        }

        /// <summary>
        /// Returns a string the represents the serialised node layer.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNodeLayer>(this)
                .Append(p => p.Nodes)
                .ToString();
        }

        /// <summary>
        /// Determines if this object equals another.
        /// </summary>
        /// <param name="that">Object to compare to.</param>
        /// <returns></returns>
        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNodeLayer>(this, that)
                .Append(p => p.Nodes)
                .AreEqual;
        }

        /// <summary>
        /// Gets hash code.
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            var hashCodeBuilder = new HashCodeBuilder()
                   .Append(Nodes);

            return hashCodeBuilder.HashCode;
        }

        /// <summary>
        /// Determines if the objects are equal.
        /// </summary>
        /// <param name="lhs">First object to compare.</param>
        /// <param name="rhs">Second object to compare.</param>
        /// <returns></returns>
        public static bool operator ==(SerialisedNodeLayer lhs, SerialisedNodeLayer rhs)
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
        public static bool operator !=(SerialisedNodeLayer lhs, SerialisedNodeLayer rhs)
        {
            return !(lhs == rhs);
        }
    }
}
