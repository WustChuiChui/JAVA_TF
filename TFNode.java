import java.util.List;

/*
* @brief: Tensor 结构体定义, 用于抽象TensorFlow服务集成
*/

public class TFNode {
    String tfName;      //tf node name
    DataType dataType;   //tf data type
    List<Integer> tfShape; // 1 * n tensor -> [1, n]; 2 * k tensor -> [2, k]
    List valueList;   //tensor reshaped to 1 *n tensor, 2 * 3 tensor [[1,2,3], [4,5,6]] -> [1,2,3,4,5,6]

    public TFNode() {}

    public TFNode(String tfName, DataType dataType, List<Integer> tfShape, List valueList) {
        this.tfName = tfName;
        this.dataType = dataType;
        this.tfShape = tfShape;
        this.valueList = valueList;
    }

    public String getTfName() { return tfName; }

    public void setTfName(String tfName) { this.tfName = tfName; }

    public DataType getDataType() { return dataType; }

    public void setDataType(DataType dataType) { this.dataType = dataType; }

    public List<Integer> getTfShape() { return tfShape; }

    public void setTfShape(List<Integer> tfShape) { this.tfShape = tfShape; }

    public List getValueList() { return valueList; }

    public void setValueList(List valueList) { this.valueList = valueList; }
}
