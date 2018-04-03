
describe('Base Test', function() {
  const assert = chai.assert;

  it('check ml', function() {
    assert(typeof navigator.ml !== 'undefined');
    assert(navigator.ml instanceof ML);
    //assert.equal(Object.getOwnPropertyDescriptor(navigator, 'ml').writable, false);
  });

  it('check getNeuralNetworkContext', function() {
    assert.equal(typeof navigator.ml.getNeuralNetworkContext, 'function');
    const nn = navigator.ml.getNeuralNetworkContext();
    assert(typeof nn !== 'undefined');
  });
});
