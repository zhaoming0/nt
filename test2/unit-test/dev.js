describe('Model Test', function() {
    const assert = chai.assert;
    const TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let nn;
  
    beforeEach(function(){
      nn = navigator.ml.getNeuralNetworkContext();
    });
  
    afterEach(function(){
      nn = undefined;
    });
  
    describe('#addOperation API', function() {
    
      it('raise error when the length of inputs is greater than 3 for "ADD" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
          model.addOperand(op);
          model.addOperand(op);
          model.addOperand(op);
          model.addOperand({type: nn.INT32});
          model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.ADD, [0, 1, 2, 3], [4]);
          });
        });
      });

      it('raise error when the length of outputs is greater than 2 for "ADD" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
          model.addOperand(op);
          model.addOperand(op);
          model.addOperand({type: nn.INT32});
          model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
          model.addOperand(op);
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.ADD, [0, 1, 2], [3, 4]);
          });
        });
      });

      it('raise error when the length of inputs is greater than 2 for "RESHAPE" operation', function() {
        return nn.createModel(options).then((model)=>{
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
          model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
          model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
          model.setOperandValue(1, new Int32Array([2, 3]));
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
          assert.throws(() => {
            model.addOperation(nn.RESHAPE, [0, 1, 2], [3]);
          });
        });
      });

      it('raise error when the length of outputs is greater than 1 for "RESHAPE" operation', function() {
        return nn.createModel(options).then((model)=>{
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
          model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
          model.setOperandValue(1, new Int32Array([2, 3]));
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
          assert.throws(() => {
            model.addOperation(nn.RESHAPE, [0, 1], [2, 3]);
          });
        });
      });

      it('raise error when the rank of input0 is 1 (not 2 or 4) for "SOFTMAX" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
          model.addOperand(op);
          model.addOperand({type: nn.FLOAT32});
          model.setOperandValue(1, new Float32Array([0.000001]));
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.SOFTMAX, [0, 1], [2]);
          });
        });
      });

      it('raise error when the rank of input0 is 3 (not 2 or 4) for "SOFTMAX" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
          model.addOperand(op);
          model.addOperand({type: nn.FLOAT32});
          model.setOperandValue(1, new Float32Array([0.000001]));
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.SOFTMAX, [0, 1], [2]);
          });
        });
      });

      it('raise error when the rank of input0 is 5 (not 2 or 4) for "SOFTMAX" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2, 2]};
          model.addOperand(op);
          model.addOperand({type: nn.FLOAT32});
          model.setOperandValue(1, new Float32Array([0.000001]));
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.SOFTMAX, [0, 1], [2]);
          });
        });
      });

      it('raise error when the length of inputs is greater than 2 for "SOFTMAX" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
          model.addOperand(op);
          model.addOperand({type: nn.FLOAT32});
          model.addOperand({type: nn.FLOAT32});
          model.setOperandValue(1, new Float32Array([0.000001]));
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.SOFTMAX, [0, 1, 2], [3]);
          });
        });
      });

      it('raise error when the length of outputs is greater than 1 for "SOFTMAX" operation', function() {
        return nn.createModel(options).then((model)=>{
          let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
          model.addOperand(op);
          model.addOperand({type: nn.FLOAT32});
          model.setOperandValue(1, new Float32Array([0.000001]));
          model.addOperand(op);
          model.addOperand(op);
          assert.throws(() => {
            model.addOperation(nn.SOFTMAX, [0, 1], [2, 3]);
          });
        });
      });

    });
  });