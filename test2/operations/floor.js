describe('Add Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [4, 1, 2];
  const nn = navigator.ml.getNeuralNetworkContext();
  const value1 = 0.5;
  it('check result', async function() {
    let operandIndex = 0;
    let model = await nn.createModel();
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
    model.addOperation(nn.FLOOR, [0], [1]);
    model.identifyInputsAndOutputs([0], [1]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();
    let execution = await compilation.createExecution();
    let inputData = new Float32Array(tensorLength);
    inputData.fill(value1);
    execution.setInput(0, inputData);
    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();
    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], Math.floor(inputData[i])));
    }
  });
});
