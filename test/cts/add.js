describe('CTS / Add test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result example 1', async function() {
    var model = await nn.createModel();
    var operandIndex = 0;

    let op1_value = [1.0, 2.0];
    let op2_value = [3.0, 4.0];
    let op3_expect = [4.0, 6.0];

    var type1 = {type: nn.INT32};
    var type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    var type0_length = product(type0.dimensions);

    var op1 = operandIndex++;
    model.addOperand(type0);
    var op2 = operandIndex++;
    model.addOperand(type0);
    var act = operandIndex++;
    model.addOperand(type1);
    var op3 = operandIndex++;
    model.addOperand(type0);

    let op2_input = new Float32Array(op2_value);
    model.setOperandValue(op2, op2_input);

    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.ADD, [op1, op2, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqual(op3_output[i], op3_expect[i]));
    }
  });
});
