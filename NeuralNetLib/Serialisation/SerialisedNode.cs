using RichTea.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib.Serialisation
{
    public class SerialisedNode
    {
        public double Bias { get; set; }
        public double[] Weights { get; set; }

        public Node CreateNode()
        {
            var node = new Node(Weights.Length)
            {
                Weights = Weights.ToArray(),
                Bias = Bias
            };
            return node;
        }

        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNode>(this)
                .Append(p => p.Bias)
                .Append(p => p.Weights)
                .ToString();
        }

        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNode>(this, that)
                .Append(p => p.Bias)
                .Append(p => p.Weights)
                .AreEqual;
        }

        public override int GetHashCode()
        {
            var hash = new HashCodeBuilder<SerialisedNode>(this)
                .Append(Bias)
                .Append(Weights)
                .HashCode;
            return hash;
        }

        public static bool operator ==(SerialisedNode lhs, SerialisedNode rhs)
        {
            if (ReferenceEquals(lhs, null))
            {
                return ReferenceEquals(rhs, null);
            }

            return lhs.Equals(rhs);
        }

        public static bool operator !=(SerialisedNode lhs, SerialisedNode rhs)
        {
            return !(lhs == rhs);
        }
    }
}
