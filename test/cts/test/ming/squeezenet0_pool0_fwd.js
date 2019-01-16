describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('squeezenet0_pool0_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let i0_value;
    let op3_expect;

    await fetch('./cts/test/ming/squeezenet0_relu0_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });

    await fetch('./cts/test/ming/squeezenet0_pool0_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3_expect = file_data;
    });

    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 111, 111, 64]};//output
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};//input
    let type0_length = product(type0.dimensions);

    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let relu6_activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(stride, new Int32Array([20]));
    model.setOperandValue(filter, new Int32Array([20]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(relu6_activation, new Int32Array([3]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, relu6_activation], [output]);

    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);

    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
