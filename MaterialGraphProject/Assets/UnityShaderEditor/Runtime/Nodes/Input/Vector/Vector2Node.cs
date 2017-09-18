using System.Collections.Generic;
using UnityEngine.Graphing;

namespace UnityEngine.MaterialGraph
{
    [Title("Input/Vector/Vector 2")]
    public class Vector2Node : AbstractMaterialNode, IGeneratesBodyCode
    {
        [SerializeField]
        private Vector2 m_Value;

        public const int OutputSlotId = 0;
        private const string kOutputSlotName = "Value";

        public Vector2Node()
        {
            name = "Vector2";
            UpdateNodeAfterDeserialization();
        }

        public sealed override void UpdateNodeAfterDeserialization()
        {
            AddSlot(new MaterialSlot(OutputSlotId, kOutputSlotName, kOutputSlotName, SlotType.Output, SlotValueType.Vector2, Vector4.zero));
            RemoveSlotsNameNotMatching(new[] { OutputSlotId });
        }

        public Vector2 value
        {
            get { return m_Value; }
            set
            {
                if (m_Value == value)
                    return;

                m_Value = value;

                if (onModified != null)
                    onModified(this, ModificationScope.Node);
            }
        }

        public override void CollectShaderProperties(PropertyCollector properties, GenerationMode generationMode)
        {
            if (!generationMode.IsPreview())
                return;

            properties.AddShaderProperty(new Vector2ShaderProperty()
            {
                name = GetVariableNameForNode(),
                generatePropertyBlock = false,
                value = value
            });
        }

        public void GenerateNodeCode(ShaderGenerator visitor, GenerationMode generationMode)
        {
            if (generationMode.IsPreview())
                return;

            visitor.AddShaderChunk(precision + " " + GetVariableNameForNode() + " = " + m_Value + ";", true);
        }

        public override string GetVariableNameForSlot(int slotId)
        {
            return GetVariableNameForNode();
        }

        public override void CollectPreviewMaterialProperties(List<PreviewProperty> properties)
        {
            properties.Add(new PreviewProperty()
            {
                m_Name = GetVariableNameForNode(),
                m_PropType = PropertyType.Vector2,
                m_Vector4 = m_Value
            });
        }
    }
}
