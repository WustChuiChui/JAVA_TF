package com.common;

import java.util.List;
import java.util.Map;

/*
* @brief: TensorFlow 通用模型fetch_dic类的定义
* @Created: 2019-03-25
*/

public class TensorOutput {
    private List<String> keyList;   //tf model fetch keys
    private Map<String, String> outputMap;   //fetched result, key in keyList, value splited by ','
                                    // [0.4, 0.3, 0.2, 0.1] -> "[0.4,0.3,0.2,0.1]"

    public TensorOutput() {}

    public TensorOutput(List<String> keyList) { this.keyList = keyList; }

    public List<String> getKeyList() { return keyList; }

    public void setKeyList(List<String> keyList) { this.keyList = keyList; }

    public Map<String, String> getOutputMap() { return outputMap; }

    public void setOutputMap(Map<String, String> outputMap) { this.outputMap = outputMap; }
}
