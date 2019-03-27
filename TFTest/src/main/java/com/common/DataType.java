package com.common;

/*
* tf dataType enum
*/

public enum DataType {
    FLOAT("float"),
    DOUBLE("double"),
    INT32("int32"),
    INT64("int64");

    private String dataType;

    DataType(String dataType) { this.dataType = dataType; }

    public String getDataType() { return this.dataType; }

    static public DataType getDataType(String dataType) {
        for (DataType tmp_type : DataType.values()) {
            if (tmp_type.getDataType().equals(dataType)) {
                return tmp_type;
            }
        }
        return null;
    }
}
