import FastaiNotebook_04_callbacks
// import Path
// import TensorFlow
// 
// let data = mnistDataBunch(flat: true)
// let (n,m) = (60000,784)
// let c = 10
// let nHid = 50
// let opt = SGD<BasicModel, Float>(learningRate: 1e-2)
// func modelInit() -> BasicModel {return BasicModel(nIn: m, nHid: nHid, nOut: c)}
// // TODO: When TF-421 is fixed, switch back to the normal `softmaxCrossEntropy`.
// 
// @differentiable(vjp: _vjpSoftmaxCrossEntropy)
// func softmaxCrossEntropy1<Scalar: TensorFlowFloatingPoint>(
//     _ features: Tensor<Scalar>, _ labels: Tensor<Scalar>
// ) -> Tensor<Scalar> {
//     return Raw.softmaxCrossEntropyWithLogits(features: features, labels: labels).loss.mean()
// }
// 
// @usableFromInline
// func _vjpSoftmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
//     features: Tensor<Scalar>, labels: Tensor<Scalar>
// ) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
//     let (loss, grad) = Raw.softmaxCrossEntropyWithLogits(features: features, labels: labels)
//     let batchSize = Tensor<Scalar>(features.shapeTensor[0])
//     return (loss.mean(), { v in ((v / batchSize) * grad, Tensor<Scalar>(0)) })
// }
// 
// //let learner = Learner(data: data, /*lossFunction: softmaxCrossEntropy1, */optimizer: opt, initializingWith: modelInit)
// 
let learner = Learner()
// //learner.delegates = [type(of: learner).TrainEvalDelegate(), type(of: learner).AvgMetric(metrics: [accuracy])]
//learner.delegates = [type(of: learner).TrainEvalDelegate()]

let foo = type(of: learner).TrainEvalDelegate()

//print("hello world")
