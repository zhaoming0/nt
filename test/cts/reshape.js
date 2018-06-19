describe('Reshape test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result example 1', async function() {
    var model = await nn.createModel();
    var operandIndex = 0;

    let op1_value = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let op3_expect = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    var type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 3, 3]};
    var type0_length = product(type0.dimensions);
    var type2 = {type: nn.TENSOR_FLOAT32, dimensions: [9]};
    var type2_length = product(type2.dimensions);
    var type1 = {type: nn.TENSOR_INT32, dimensions: [1]};
    var type1_length = product(type1.dimensions);

    var op1 = operandIndex++;
    model.addOperand(type0);
    var op2 = operandIndex++;
    model.addOperand(type1);
    var op3 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Int32Array([-1]));
    model.addOperation(nn.RESHAPE, [op1, op2], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type2_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqual(op3_output[i], op3_expect[i]));
    }
  });
});
