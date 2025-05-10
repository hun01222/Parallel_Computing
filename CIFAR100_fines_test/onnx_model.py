import onnx

model = onnx.load("resnet-18_0.onnx")

print("== Inputs ==")
for i in model.graph.input:
    print(f"- {i.name}: {i.type}")

print("\n== Outputs ==")
for o in model.graph.output:
    print(f"- {o.name}: {o.type}")

print("\n== Nodes ==")
for n in model.graph.node[:10]:  # 너무 길면 앞부분만
    print(f"- {n.op_type}: {n.name} (inputs={n.input}, outputs={n.output})")

print("\n== Initializers ==")
for init in model.graph.initializer[:10]:  # 일부만 출력
    print(f"- {init.name}, shape={init.dims}")

print("\n== Full Graph ==")
print(onnx.helper.printable_graph(model.graph))
