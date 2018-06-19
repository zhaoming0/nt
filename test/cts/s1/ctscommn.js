
function runCTSTest(testName, modleFile, exampleFile) {
  var modelJson;
  var exampleJson;

  $.ajaxSettings.async = false; 
  //model.Operation("ADD", i1, i2, act).To(i3)
  $.getJSON("cts/" + modleFile, function(json){
    modelJson = json;
  });
 
  $.getJSON("cts/" + exampleFile, function(json){
    exampleJson = json;
  });

  describe(testName, function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
    let i = 0;
    exampleJson.test.forEach(async example => {
       i++;
      //TODO: test name 
      it('check result ' + i, async function() {
        let model = await nn.createModel(options);
        let lenArr = new Array();
        
        for (opIndex in modelJson.inputs) {
          var opType = modelJson.inputs[opIndex].type;
          if (opType == "TENSOR_FLOAT32") {
            //TODO: scale, zeroPoint
            op = {type: nn.TENSOR_FLOAT32, dimensions: modelJson.inputs[opIndex].dimensions};
            lenArr[opIndex] = product(op.dimensions);
            model.addOperand(op);
          } else if (opType == "INT32") {
            lenArr[opIndex] = 0;
            model.addOperand({type: nn.INT32});
            model.setOperandValue(opIndex, new Int32Array([modelJson.inputs[opIndex].value]));
          }
        }

        for (outOpIndex in modelJson.outputs) {
          var opType = modelJson.outputs[outOpIndex].type;
          if (opType == "TENSOR_FLOAT32") {
            outOp = {type: nn.TENSOR_FLOAT32, dimensions: modelJson.outputs[outOpIndex].dimensions};
            lenArr[Number(opIndex) + 1 + Number(outOpIndex)] = product(outOp.dimensions);
            //TODO: scale, zeroPoint
            model.addOperand(outOp);
          } 
        }

        // for (eOpIndex in example.inputs) {
        //   if (eOpIndex == 0) {
        //     var inputData0 = new Float32Array(lenArr[eOpIndex]);
        //     inputData0.set(example.inputs[eOpIndex])
        //   } else {
        //     //map Float32Array
        //     let inputData = new Float32Array(lenArr[eOpIndex]);
        //     inputData.set(example.inputs[eOpIndex])
        //     model.setOperandValue(eOpIndex, inputData);
        //   }
        // }
        //TODO: map str -> operation
        model.addOperation(nn.ADD, modelJson.operation.inputs, modelJson.operation.outputs);
        model.identifyInputsAndOutputs(modelJson.identifyInputs, modelJson.identifyOutputs);
        
        await model.finish();
    
        let compilation = await model.createCompilation();
    
        compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    
        await compilation.finish();
    
        let execution = await compilation.createExecution();

        let j = 0;
        for (inOpIndex in modelJson.identifyInputs) {
          var inputData = new Float32Array(lenArr[inOpIndex]);
          inputData.set(example.inputs[j])
          execution.setInput(Number(inOpIndex), inputData);
        }

        outTensorLength = lenArr[modelJson.identifyOutputs[0]];
        let outputData = new Float32Array(outTensorLength);
        execution.setOutput(0, outputData);
    
        await execution.startCompute();

        expectedData = new Float32Array(outTensorLength);
        expectedData.set(example.outputs[0]);

        for (let i = 0; i < outTensorLength; ++i) {
          assert.isTrue(almostEqual(outputData[i], expectedData[i])); 
        }
      });
    });

  });
}
