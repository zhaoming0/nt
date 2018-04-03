describe('Model Test', function() {
  const assert = chai.assert;
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  }); 

  it('check addOperand is a function', function() {
    return nn.createModel().then((model)=>{
      assert.equal(typeof model.addOperand, 'function');
    });
  });

  it('verify return value of addOperand is void', function() {
    return nn.createModel().then((model)=>{
      assert.equal(model.addOperand({type: nn.INT32}), undefined);
    });
  });

  it('verify addOperand with operand of "FLOAT32" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.FLOAT32});
      });
    });
  });

  it('verify addOperand with operand of "INT32" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.INT32});
      });
    });
  });

  it('verify addOperand with operand of "UINT32" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.UINT32});
      });
    });
  });

  it('verify addOperand with operand of "TENSOR_FLOAT32" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.TENSOR_FLOAT32});
      });
    });
  });

  it('verify addOperand with operand of "TENSOR_INT32" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.TENSOR_INT32});
      });
    });
  });

  it('verify addOperand with operand of "TENSOR_QUANT8_ASYMM" type is ok', function() {
    return nn.createModel().then((model)=>{
      assert.doesNotThrow(() => {
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM});
      });
    });
  });

  it('addOperand raise error when operand type is not in 0~5', function() {
    return nn.createModel().then((model)=>{
      assert.throws(() => {
        model.addOperand({type: 6});
      });
    });
  });

  it('addOperand raise error when passing an operand without type', function() {
    return nn.createModel().then((model)=>{
      assert.throws(model.addOperand({dimensions: [2,2,2,2]}));
    });
  });

  it('addOperand raise error when no parameter', function() {
    return nn.createModel().then((model)=>{
      assert.throws(() => {
        model.addOperand();
      });
    });
  });

  it('addOperand raise error when passing an invalid parameter', function() {
    return nn.createModel().then((model)=>{
      assert.throws(() => {
        model.addOperand(123);
      });
    });
  });

  it('addOperand raise error when passing two parameters', function() {
    return nn.createModel().then((model)=>{
      assert.throws(() => {
        model.addOperand({type: nn.INT32}, {type: nn.INT32});
      });
    });
  });

  it('check setOperandValue is a function', function() {
    return nn.createModel().then((model)=>{
      assert.equal(typeof model.setOperandValue, 'function');
    });
  });

  it('check return value of setOperandValue is void', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.equal(model.setOperandValue(0, new Int32Array([nn.FUSED_NONE])), undefined);
    });
  });

  it('when operand type is "FLOAT32", passing a Float32Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.FLOAT32});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Float32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('when operand type is "TENSOR_FLOAT32", passing a Float32Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_FLOAT32});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Float32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('when operand type is "INT32", passing an Int32Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('when operand type is "TENSOR_INT32", passing an Int32Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_INT32});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('when operand type is "UINT32", passing an Uint32Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.UINT32});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Uint32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('when operand type is "TENSOR_QUANT8_ASYMM", passing an Uint8Arry as sencond parameter of setOperandValue is ok', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_QUANT8_ASYMM});
      assert.doesNotThrow(() => {
        model.setOperandValue(0, new Uint8Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when no parameter', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.throws(() => {
        model.setOperandValue();
      });
    });
  });

  it.skip('setOperandValue raise error when first parameter is negative', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.throws(() => {
        model.setOperandValue(-1, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it.skip('setOperandValue raise error when first parameter is larger than the size of operands', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.throws(() => {
        model.setOperandValue(1, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is FLOAT32 and its second parameter as Int32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.FLOAT32});
      assert.throws(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is TENSOR_FLOAT32 and its second parameter as Int32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_FLOAT32});
      assert.throws(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is INT32 and its second parameter as Float32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.INT32});
      assert.throws(() => {
        model.setOperandValue(0, new Float32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is TENSOR_INT32 and its second parameter as Float32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_INT32});
      assert.throws(() => {
        model.setOperandValue(0, new Float32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is UINT32 and its second parameter as Int32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.UINT32});
      assert.throws(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

  it('setOperandValue raise error when operand type is TENSOR_QUANT8_ASYMM and its second parameter as Int32Array', function() {
    return nn.createModel().then((model)=>{
      model.addOperand({type: nn.TENSOR_QUANT8_ASYMM});
      assert.throws(() => {
        model.setOperandValue(0, new Int32Array([nn.FUSED_NONE]));
      });
    });
  });

/*
  it.skip('addOperation', function() {
    assert.equal(typeof model.addOperation, 'function');
  });

  it.skip('identifyInputsAndOutputs', function() {
    assert.equal(typeof model.identifyInputsAndOutputs, 'function');
  });

  it.skip('finish', function() {
    assert.equal(typeof model.finish, 'function');
  });

  it.skip('createCompilation', function() {
    assert.equal(typeof model.createCompilation, 'function');
  });*/
});
