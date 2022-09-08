import onnx

model = onnx.load("bert_large_v1_1_fake_quant.onnx")
graph = model.graph
input0 =  graph.input[0]
input1 =  graph.input[1]
input2 =  graph.input[2]
input0.type.tensor_type.elem_type = 6 #int32
input1.type.tensor_type.elem_type = 6
input2.type.tensor_type.elem_type = 6
graph.input.remove(graph.input[2])
graph.input.remove(graph.input[1])
graph.input.remove(graph.input[0])
graph.input.extend([input0])
graph.input.extend([input1])
graph.input.extend([input2])
for n in graph.node:
    if "Gather" in n.name:
        print(n)
onnx.save(model, "bert_large_v1_1_fake_quant_int32.onnx")
