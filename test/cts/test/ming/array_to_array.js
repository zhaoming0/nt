let fs = require('fs');
// let output = []
// let inputs = fs.readFileSync('squeezenet0_conv0_weight.txt');
let inputs = fs.readFileSync('a.txt');
let view = new Float32Array(inputs);
console.log(Array.isArray(view))

// function fun(arr) {
//   for (let i=0;i<arr.length;i++) {
//     if (Array.isArray(arr[i])) {
//       fun(arr[i]);
//     }else {
//       console.log(arr[i])
//       output.push(arr[i]);
//     }
//   }
// }
// fun(arrs);

Array.prototype.deepFlatten = function() {
      var result = []; //定义保存结果的数组
      this.forEach(function(val, idx) { //遍历数组
        if (Array.isArray(val)) { //判断是否为子数组
          val.forEach(arguments.callee); //为子数组则递归执行
        } else {
          result.push(val); //不为子数组则将值存入结果数组中
        }
      });
      return result; //返回result数组
}
var arr = [2, 3, [2, 2],
      [3, 'f', ['w', 3]], { "name": 'Tom' }
];
// console.log(view)
// console.log(view.deepFlatten())




// fs.writeFile('testa.txt',output, function(err) {
//   if (err) {
//     console.log(err);
//   }
// })
//

//let strs = inputs.toString();
//strs.replace("[", "")

