package com.xiaomi.ai.domain.hotel.tf;

import java.util.List;

/*
* @brief: TensorFlow 模型通用输入类(feed_dic)的定义
* @Created: 2019-03-25
*/

public class TensorInput {
    private List<TFNode> tfNodeList;   // for feed_dic nodes

    public TensorInput() {}

    public TensorInput(List<TFNode> tfNodeList) { this.tfNodeList = tfNodeList; }

    public List<TFNode> getTfNodeList() { return tfNodeList; }

    public void setTfNodeList(List<TFNode> tfNodeList) { this.tfNodeList = tfNodeList; }
}
