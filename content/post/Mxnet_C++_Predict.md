---
title: "Mxnet的C++推断接口的使用"
date: 2018-06-04
draft: true
markup: mmark
---
Mxnet提供了比较简单清晰的C++推断接口，下面详细说明一下，并且用上篇的情感分析模型来测试一下。
## 1. 主要函数
### 1.1 BufferFile类
BUfferFile类是用来读取训练完的参数文件和网络文件的，并且还能够读取Mxnet保存的NDArray文件，其定义在Mxnet的示例中
``` c++
class BufferFile {
public:
	string file_path_;
	int length_;
	char* buffer_;

	explicit BufferFile(string file_path)
		:file_path_(file_path) {

		ifstream ifs(file_path.c_str(), ios::in | ios::binary);
		if (!ifs) {
			cerr << "Can't open the file. Please check " << file_path << ". \n";
			length_ = 0;
			buffer_ = NULL;
			return;
		}

		ifs.seekg(0, ios::end);
		length_ = ifs.tellg();
		ifs.seekg(0, ios::beg);
		cout << file_path.c_str() << " ... " << length_ << " bytes\n";

		buffer_ = new char[sizeof(char) * length_];
		ifs.read(buffer_, length_);
		ifs.close();
	}

	int GetLength() {
		return length_;
	}
	char* GetBuffer() {
		return buffer_;
	}

	~BufferFile() {
		if (buffer_) {
			delete[] buffer_;
			buffer_ = NULL;
		}
	}
};
```
### 1.2 MXPredCreate
MXPredCreate是用来创建与Mxnet交互的Handle，通过这个Handle可以将数据送入网络并获取结果。
``` C++
MXPredCreate (
    const char * symbol_json_st, 
    const void * param_bytes, 
    int param_size, 
    int dev_type, 
    int dev_id, 
    mx_uint num_input_nodes, 
    const char ** input_keys, 
    const mx_uint * input_shape_indptr, 
    const mx_uint * input_shape_data, 
    PredictorHandle * out )
```
参数
- **symbol_json_str**  The JSON string of the symbol.   
- **param_bytes**  The in-memory raw bytes of parameter ndarray file.   
- **param_size**  The size of parameter ndarray file.   
- **dev_type**  The device type, 1: cpu, 2:gpu   
- **dev_id**  The device id of the predictor.   
- **num_input_nodes**  Number of input nodes to the net, For feedforward net, this is 1.   
- **input_keys**  The name of input argument. For feedforward net, this is {"data"}   
- **input_shape_indptr**  Index pointer of shapes of each input node. The length of this array = num_input_nodes + 1. For feedforward net that takes 4 dimensional input, this is {0, 4}.   
- **input_shape_data**  A flatted data of shapes of each input node. For feedforward net that takes 4 dimensional input, this is the shape data.   
- **out**  The created predictor handle.  

第一个和第二个为网络JSON内容和参数内容  
`dev_type` 为设备类型，1是CPU，2是GPU  
`dev_id` 是设备号，也就是说用哪个设备，如果你有很多个GPU，可以指定在某个GPU上。  
`num_input_nodes` 是数据输入的个数，也就是说你的网络是但输入还是多输入  
`input_keys` 是说明你的输入参数名称，默认的应该是data，当然你也可以指定，这是在训练的时候指定的  
`input_shape_data` 是数据输入的尺寸，这和在网络定义的时候是一样的。  
`input_shape_indptr` 是输入数据的指针索引，用来说明`input_shape_data`中的各个维度与输入的对应关系，比如如果数据的尺寸是
``` c++
const mx_uint input_shape_data[4] = {1,3,224,224};
```
则`input_shape_indptr`则应该为
``` c++
const mx_uint input_shape_indptr[3] = {
    0,  // 输入的尺寸信息从input_shape_data的第0个位置开始, 到4结束: 1, 3, 224, 224
    4};
```
如果是多个数据输入，比如网络需要两个输入，`input_shape_data`为
``` c++
const mx_uint input_shape_data[8] = {
    1, 3, 224, 224,    // 第一个输入的大小
    1, 3, 224, 224     // 第二个输入的大小
    };
```
则`input_shape_indptr`则应该为
``` c++
const mx_uint input_shape_indptr[3] = {
    0,  // 第一个输入的尺寸信息从input_shape_data的第0个位置开始, 到4结束: 1, 3, 224, 224
    4,  // 第二个输入的尺寸信息从inptu_shape_data的第4个位置开始, 到8结束: 1, 3, 224, 224
    8};
```
`out`就是创建的Handle
### 1.3 把数据送入网络计算
把数据送入网络
``` c++
MXPredSetInput (
    PredictorHandle handle, 
    const char * key, 
    const mx_float * data, 
    mx_uint size )
```
参数分别是  
- **handle** 用于和MXnet交互的Handle  
- **key** 数据输入参数的名称  
- **data** 需要输入的数据
- **size** 数据大小  

前向传播计算结果
``` c++
MXPredForward (PredictorHandle handle)
```
### 1.4 获取网络计算结果
获取输出的尺寸
``` c++
MXPredGetOutputShape (
    PredictorHandle handle, 
    mx_uint index, 
    mx_uint ** shape_data, 
    mx_uint * shape_ndim )
```
参数分别是
- **handle**
- **index** 网络输出的节点的索引，只有一个的话就设为0 
- **shape_data** 用于保存数据的形状
- **shape_ndin** 用于保存数据的尺寸  

获取网络输出的结果
``` c++
MXPredGetOutput (
    PredictorHandle handle, 
    mx_uint index, 
    mx_float * data, 
    mx_uint size )
```
参数分别
- **handle**
- **index** 输出节点的索引和`MXPredGetOutputShape`的一样，只有一个的话就设为0  
- **data** 获取的数据
- **size** 数据的大小  

## 2. 使用C++接口来进行情感分析的预测
### 2.1 创建Hadnle
``` c++
string json_file = "net-symbol.json";
string param_file = "net-0000.params";
BufferFile json_data(model_json);
BufferFile param_data(model_params);

mx_uint num_input_nodes = 1;
const char* input_key[1] = { "data" };
const char** input_keys = input_key;

int input_shape[1] = { 100 };
const mx_uint input_shape_indptr[2] = { 0,2 };
const mx_uint input_shape_data[2] = { 1,static_cast<mx_uint>(input_shape[0]) };
input_data_size = input_shape[0];
PredictorHandle pred_hnd = 0;

int dev_type = 1;
int dev_id = 1;
MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
    static_cast<const char*>(param_data.GetBuffer()),
    static_cast<int>(param_data.GetLength()),
    dev_type,
    dev_id,
    num_input_nodes,
    input_keys,
    input_shape_indptr,
    input_shape_data,
    &pred_hnd);
assert(pred_hnd);
```
### 2.2 输入送入网络并获取数据
``` c++
MXPredSetInput(pred_hnd, "data", data.data(), input_data_size);
MXPredForward(pred_hnd);
mx_uint output_index = 0;

mx_uint* shape = nullptr;
mx_uint shape_len;

// Get Output Result
MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

size_t size = 1;
for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

vector<float> output(size);

MXPredGetOutput(pred_hnd, output_index, &(output[0]), static_cast<mx_uint>(size));
```
### 2.3 辅助代码
将网络从gluon导出
``` python
net.export('net')
```
把数据保存成csv
``` python
f = open('test.csv','w',encoding='utf-8')
a = x_test[0].asnumpy()
s = ','.join([str(x) for x in a])
f.write(s)
f.close()
```
c++中读取csv文件的内容
``` c++
ifstream fin("test.csv");
string line;
vector<float> data;
while (getline(fin, line)) {

    istringstream sin(line);
    string field;
    while (getline(sin, field, ',')) {
        data.push_back(atof(field.c_str()));
    }
}
fin.close();
```
读取Python中保存的NDArray数据
``` c++
string nd_file = "test.nd";

const mx_float* nd_data = nullptr;
NDListHandle nd_hnd = nullptr;
BufferFile nd_buf(nd_file);

if (nd_buf.GetLength() > 0) {
    mx_uint nd_index = 0;
    mx_uint nd_len;
    const mx_uint* nd_shape = nullptr;
    const char* nd_key = nullptr;
    mx_uint nd_ndim = 0;

    MXNDListCreate(static_cast<const char*>(nd_buf.GetBuffer()),
        static_cast<int>(nd_buf.GetLength()),
        &nd_hnd, &nd_len);

    MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
}
```