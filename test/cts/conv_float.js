describe('Conv float test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result example 1', async function() {
    var model = await nn.createModel();
    var operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    var type3 = {type: nn.INT32};
    var type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    var type1_length = product(type1.dimensions);
    var type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    var type0_length = product(type0.dimensions);
    var type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    var type2_length = product(type2.dimensions);

    var op1 = operandIndex++;
    model.addOperand(type0);
    var op2 = operandIndex++;
    model.addOperand(type1);
    var op3 = operandIndex++;
    model.addOperand(type2);
    var pad0 = operandIndex++;
    model.addOperand(type3);
    var act = operandIndex++;
    model.addOperand(type3);
    var stride = operandIndex++;
    model.addOperand(type3);
    var op4 = operandIndex++;
    model.addOperand(type1);

    model.setOperandValue(op2, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    model.setOperandValue(op3, new Float32Array([0]));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqual(op4_output[i], op4_expect[i]));
    }
  });
});
