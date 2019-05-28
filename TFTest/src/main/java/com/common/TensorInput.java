package com.common;

import java.util.List;

/*
* @brief: TensorFlow 模型通用输入类(feed_dic)的定义
*/

public class TensorInput {
    private String requestId;
    private List<TFNode> tfNodeList;   // for feed_dic nodes

    public TensorInput() {}

    public TensorInput(List<TFNode> tfNodeList) { this.tfNodeList = tfNodeList; }

    public TensorInput(String requestId, List<TFNode> tfNodeList) {
        this.requestId = requestId;
        this.tfNodeList = tfNodeList;
    }

    public String getRequestId() { return requestId; }

    public void setRequestId(String requestId) { this.requestId = requestId; }

    public List<TFNode> getTfNodeList() { return tfNodeList; }

    public void setTfNodeList(List<TFNode> tfNodeList) { this.tfNodeList = tfNodeList; }
}
