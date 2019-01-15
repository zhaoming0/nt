describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Max pool float example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op3_expect;

    await fetch('./cts/test/ming/squeezenet0_relu25_fwd').then((res) => {
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

    await fetch('./cts/test/ming/squeezenet0_pool3_fwd').then((res) => {
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
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 1000]};
    let type0_length = product(type0.dimensions);
    let oupt1 = {type: nn.INT32};
    let oupt0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1000]};
    let oupt0_length = product(oupt0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let cons1 = operandIndex++;
    model.addOperand(type1);
    let pad0 = operandIndex++;
    model.addOperand(type1);
    let act = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(oupt0);

    model.setOperandValue(cons1, new Int32Array([1]));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.MAX_POOL_2D, [op1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(oupt0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
