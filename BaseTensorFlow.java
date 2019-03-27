import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
* @brief: 通用TensorFlow模型调用类的定义
* @Created: 2019-03-25
*/
public class BaseTensorFlow {
    Session session = null;
    Graph graph = null;
    private Logger LOGGER = LoggerFactory.getLogger(BaseTensorFlow.class);

    private  BaseTensorFlow baseTensorFlow = null;

    public static BaseTensorFlow getInstance(String modelPath) {
        BaseTensorFlow baseTensorFlow = new BaseTensorFlow();
        if (baseTensorFlow.init(modelPath) != 0) {
            return null;
        }
        return baseTensorFlow;
    }

    /*
    * @brief: 导入TF模型
    * @param[in]: modelPath tf模型文件路径,文件格式应为pb格式
    * @return: int success 0; else -1
    */
    private int init(String modelPath) {
        InputStream inputStream = null;
        try {
            this.graph = new Graph();
            try {
                inputStream = new FileInputStream(modelPath);
            } catch (IOException e) {
                LOGGER.error("load model failed. " + e.getMessage());
                return -1;
            }
            try {
                this.graph.importGraphDef(IOUtils.toByteArray(inputStream));
                this.session = new Session(graph);
            } catch (Exception e) {
                LOGGER.error("session init failed. " + e);
                return -1;
            } finally {
                if (inputStream != null) {
                    inputStream.close();
                }
            }
        } catch (Exception e) {
            LOGGER.error("graph init failed. " + e);
            return -1;
        }
        LOGGER.info("tensorflow init success. " + modelPath);
        return 0;
    }

    /*
    * @brief: 外部调用接口,获取模型调用结果
    * @param[in]: tensorInput 模型输入参数对象
    * @param[out]: tensorOutput 模型输出结果对象
    * @return: int success 0; else -1
    */
    public int run(TensorInput tensorInput, TensorOutput tensorOutput) {
        if (tensorInput == null || tensorOutput == null) {
            LOGGER.warn("base tensorflow model get null input.");
            return -1;
        }
        List<TFNode> tfNodes = tensorInput.getTfNodeList();
        try {
            Session.Runner runner = this.session.runner();
            for (int idx = 0; idx < tfNodes.size(); ++idx) {
                TFNode node = tfNodes.get(idx);
                if (node.dataType == null || StringUtils.isEmpty(node.tfName)) {
                    LOGGER.warn("invalid input tensor found. " + node.tfName);
                    return -1;
                }
                Tensor tensor = buildTensor(node.valueList, node.dataType, node.tfShape);
                if (tensor == null) {
                    LOGGER.error("input tensor is null. " + node.tfName);
                }
                runner.feed(node.tfName, tensor);
            }
            List<String> keyList = tensorOutput.getKeyList();
            for (int idx = 0; idx < keyList.size(); ++idx) {
                runner.fetch(keyList.get(idx));
            }

            List<Tensor<?>> res = runner.run();
            tensorOutput.setOutputMap(parseTensorResult(res, keyList));

            //free momery
            for (int idx = 0; idx < res.size(); ++idx) {
                res.get(idx).close();
            }
        } catch (Exception e) {
            LOGGER.warn("session run failed. " + e);
            return  -1;
        }
        return 0;
    }

    /*
    * @brief: 将Tensor node 转成TensorFlow内部调用使用的tensor
    * @param[in]: values, tensor node 的values
    * @param[in]: dataType, tensor的类型描述对象
    * @param[in]: shapeList, tensor的shape
    * @return: tensor object or null
    */
    private Tensor buildTensor(List values, DataType dataType, List<Integer> shapeList) {
        long[] bufferShape = shapeList.stream().mapToLong(Integer::longValue).toArray();
        String valueType = dataType.getDataType();
        if (valueType.equals("int32") || valueType.equals("int64")) {
            IntBuffer buffer = IntBuffer.allocate(values.size());
            for (int index = 0; index < values.size(); ++index) {
                buffer.put((Integer) values.get(index));
            }
            buffer.flip();
            return Tensor.create(bufferShape, buffer);
        } else if (valueType.equals("float")) {
            FloatBuffer buffer = FloatBuffer.allocate(values.size());
            for (int index = 0; index < values.size(); ++index) {
                buffer.put((Float) values.get(index));
            }
            buffer.flip();
            return Tensor.create(bufferShape, buffer);
        } else if (valueType.equals("double")) {
            DoubleBuffer buffer = DoubleBuffer.allocate(values.size());
            for (int index = 0; index < values.size(); ++index) {
                buffer.put((Double) values.get(index));
            }
            buffer.flip();
            return Tensor.create(bufferShape, buffer);
        }
        LOGGER.warn("int32, int64, float and double were supported. invalid dataType with " + valueType);
        return null;
    }

    /*
    * @brief: 解析TensorFlow模型运行结果,将其值按key,value存入map
    * @param[in]: tensorList TF模型fetch的Tensor列表
    * @param[in]: keyList TF模型fetch的keyList
    * @return： map<String, String> Tensor解析后结果的存储对象
    */
    private Map<String, String> parseTensorResult(List<Tensor<?>> tensorList, List<String> keyList) {
        Map<String, String> result = new HashMap<>();
        Integer tensorNum = tensorList.size();
        Integer keyNum = keyList.size();
        if (tensorList.isEmpty() || keyList.isEmpty() || tensorNum != keyNum) {
            LOGGER.warn("keyList and fetch tensor were not match");
            return result;
        }
        for (int idx = 0; idx < tensorNum; ++idx) {
            Tensor tensor = tensorList.get(idx);
            Integer outSize = getTensorSize(tensor);
            //对于float等类型的Tensor,其返回结果直接转成Float, outsize为0
            if (outSize <=0) {
                result.put(keyList.get(idx), Float.toString(tensor.floatValue()));
                continue;
            }
            FloatBuffer outArray = FloatBuffer.allocate(outSize);
            tensor.writeTo(outArray);
            outArray.flip();
            StringBuffer  outStr = new StringBuffer();
            while (outArray.hasRemaining()) {
                outStr.append(Float.toString(outArray.get()) + ",");
            }
            outStr = outStr.deleteCharAt(outStr.length() - 1);
            result.put(keyList.get(idx), "[" + outStr.toString() + "]");
        }
        return result;
    }

    /*
    * @brief: 求Tensor对象的shape大小
    * @param[in]: tensor TF模型输出结果对象
    * @param[out]: Integer tensor的大小
    */
    private Integer getTensorSize(Tensor tensor) {
        long[] shapes = tensor.shape();
        if (shapes.length <= 0) {
            return 0;
        }
        Integer res = 1;
        for (int idx = 0; idx < shapes.length; ++idx) {
            res *= Integer.valueOf((int)shapes[idx]);
        }
        return res;
    }
}
