describe('[CTS] Add Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2];
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result', async function() {
    let operandIndex = 0;
    let model = await nn.createModel(options);
    const tensorType = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 2.0, zeroPoint: 0};
    const tensorLength = product(tensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(tensorType);
    let input0Data = new Uint8Array(tensorLength);
    input0Data.set([1, 2]);
    model.setOperandValue(input0, input0Data);

    const tensorType1 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 1.0, zeroPoint: 0};
    const tensorLength1 = product(tensorType1.dimensions);   
    let input1 = operandIndex++;
    model.addOperand(tensorType1);
    let output = operandIndex++;
    model.addOperand(tensorType1);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);

    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Uint8Array(tensorLength1);
    input1Data.set([3, 4]);

    execution.setInput(0, input1Data);

    let outputData = new Uint8Array(tensorLength1);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], input0Data[i] * 2 + input1Data[i]));
    }
  });
});
