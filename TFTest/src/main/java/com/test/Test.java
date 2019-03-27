package com.test;
import com.common.*;
import java.util.*;

public class Test {
    public static void main(String[] args) throws Exception {
        List<Integer> tfShape = Arrays.asList(1, 3);
        List valueList = Arrays.asList(1.0f, 2.0f, 3.0f);
        DataType dataType = DataType.getDataType("float");
        TFNode tfNode_1 = new TFNode("x", dataType, tfShape, valueList);

        valueList = null;
        valueList = Arrays.asList(2.0f, 3.0f, 4.0f);
        TFNode tfNode_2 = new TFNode("y", dataType, tfShape, valueList);

        BaseTensorFlow model = BaseTensorFlow.getInstance("~/test/tf/model.pb");

        List<TFNode> tfNodes = Arrays.asList(tfNode_1, tfNode_2);
        TensorInput tensorInput = new TensorInput(tfNodes);

        List<String> keyList = Arrays.asList("z", "s");
        TensorOutput tensorOutput = new TensorOutput(keyList);

        for (int idx = 0; idx <1; ++idx) {
            int flag = model.run(tensorInput, tensorOutput);
            if (flag == -1) {
                System.out.println("model run failed.");
            } else {
                Map<String, String> output = tensorOutput.getOutputMap();
                Iterator<Map.Entry<String, String>> entries = output.entrySet().iterator();
                while (entries.hasNext()) {
                    Map.Entry<String, String> entry = entries.next();
                    System.out.println("Key= " + entry.getKey() + ", Value=" + entry.getValue());
                }
            }
        }

    }
}
