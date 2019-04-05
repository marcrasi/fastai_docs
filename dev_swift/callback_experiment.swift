
import TensorFlow

struct DataBatch {
    // Simplifying assumption: Model inputs and outputs are Tensor<Float>
    var xb: Tensor<Float>
    var yb: Tensor<Float>
}

struct Data {
    // Simplifying assumption: Batches are in an array.
    var trainBatches: [DataBatch]
}

enum CallbackEvent {
    // I haven't implemented all the events.
    case beginFit
    case beginEpoch
    case beginBatch
    case afterForwardsBackwards
}

class Callback<Opt: Optimizer>
where Opt.Model.CotangentVector == Opt.Model.AllDifferentiableVariables,
      Opt.Model.Input == Tensor<Float>,
      Opt.Model.Output == Tensor<Float> {
    func apply(event: CallbackEvent, learner: Learner<Opt>) {}
}

class Learner<Opt: Optimizer>
where Opt.Model.CotangentVector == Opt.Model.AllDifferentiableVariables,
      Opt.Model.Input == Tensor<Float>,
      Opt.Model.Output == Tensor<Float>
{
    typealias Model = Opt.Model
    var model: Model

    // (inputs, labels) -> loss
    typealias LossFunc = @differentiable (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    var lossFunc: LossFunc
    
    var optimizer: Opt
    var data: Data
    var callbacks: [Callback<Opt>]
    
    var loss: Tensor<Float> = Tensor(0)
    var grad: Model.AllDifferentiableVariables = Model.AllDifferentiableVariables.zero
    
    var epoch: Int = 0
    var epochs: Int = 0
    
    init(
        model: Model,
        lossFunc: @escaping LossFunc,
        optimizer: Opt,
        data: Data,
        callbacks: [Callback<Opt>]
    ) {
        self.model = model
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.data = data
        self.callbacks = callbacks
    }
    
    func trainOneBatch(xb: Tensor<Float>, yb: Tensor<Float>) {
        runCallbacks(event: .beginBatch)
        let context = Context(learningPhase: .training)
        // Take derivative wrt model and labels to workaround temporary AD limitation.
        let lossWithGradient = model.valueWithGradient(at: yb) { (model, yb) -> Tensor<Float> in
            let y = model.applied(to: xb, in: context)
            return lossFunc(y, yb)
        }
        self.loss = lossWithGradient.value
        self.grad = lossWithGradient.gradient.0
        runCallbacks(event: .afterForwardsBackwards)
        optimizer.update(&model.allDifferentiableVariables, along: self.grad)
    }
    
    func trainOneEpoch() {
        runCallbacks(event: .beginEpoch)
        for batch in self.data.trainBatches {
            trainOneBatch(xb: batch.xb, yb: batch.yb)
        }
    }

    func fit(epochs: Int) {
        // I haven't implemented validation.
        self.epochs = epochs
        runCallbacks(event: .beginFit)
        for epoch in 1...epochs {
            self.epoch = epoch
            trainOneEpoch()
        }
    }
    
    private func runCallbacks(event: CallbackEvent) {
        for callback in callbacks {
            callback.apply(event: event, learner: self)
        }
    }
}

// %include "EnableIPythonDisplay.swift"
// let plt = Python.import("matplotlib.pyplot")
// IPythonDisplay.shell.enable_matplotlib("inline")

class Recorder<Opt: Optimizer> : Callback<Opt>
// Hmm, this boilerplate is kind of annoying.
where Opt.Model.CotangentVector == Opt.Model.AllDifferentiableVariables,
      Opt.Model.Input == Tensor<Float>,
      Opt.Model.Output == Tensor<Float>,
      // Notice that we can add constraints so that this callback only works with certain types of learners.
      // Here, we require that the optimizer's scalar type is float so that `plt.plot` understands the
      // learning rate.
      Opt.Scalar == Float {
          
    var losses: [Float] = []
    var lrs: [Float] = []
          
    override func apply(event: CallbackEvent, learner: Learner<Opt>) {
        switch event {
        case .beginFit:
            losses = []
            lrs = []
        case .afterForwardsBackwards:
            losses.append(learner.loss.scalar!)
            lrs.append(learner.optimizer.learningRate)
        default: break
        }
    }
          
    // func plotLosses() {
    //     plt.plot(losses)
    // }
    //       
    // func plotLrs() {
    //     plt.plot(lrs)
    // }
}

class ParamScheduler<Opt: Optimizer, Param> : Callback<Opt>
// Hmm, this boilerplate is kind of annoying.
where Opt.Model.CotangentVector == Opt.Model.AllDifferentiableVariables,
      Opt.Model.Input == Tensor<Float>,
      Opt.Model.Output == Tensor<Float> {
    
    let paramKeyPath: ReferenceWritableKeyPath<Learner<Opt>, Param>
    let schedule: (Float) -> Param
    
    init(paramKeyPath: ReferenceWritableKeyPath<Learner<Opt>, Param>, schedule: @escaping (Float) -> Param) {
        self.paramKeyPath = paramKeyPath
        self.schedule = schedule
    }
          
    override func apply(event: CallbackEvent, learner: Learner<Opt>) {
        switch event {
        case .beginBatch:
            learner[keyPath: paramKeyPath] = schedule(Float(learner.epoch) / Float(learner.epochs))
        default: break
        }
    }
}

// Sum of the two inputs is the output.
let data = Data(trainBatches: [
    DataBatch(xb: [[0, 1], [2, 3]], yb: [[1], [5]]),
    DataBatch(xb: [[-3, 4], [-10, 2]], yb: [[1], [-8]]),
])

struct SillyModel : Layer {
    var dense: Dense<Float> = Dense(inputSize: 2, outputSize: 1)
    
    // A non-trained parameter to help illustrate the parameter scheduler.
    @noDerivative var sillyExtraBiasParam: Float = 0
    
    @differentiable
    func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        return dense.applied(to: input, in: context) + sillyExtraBiasParam
    }
}

@differentiable
func meanSquaredError(predictions: Tensor<Float>, labels: Tensor<Float>) -> Tensor<Float> {
    return (predictions - labels).squared().mean()
}

let model = SillyModel()

// Some typealiases to reduce repeatedly typing types.
typealias MyOptimizer = SGD<SillyModel, Float>
typealias MyLearner = Learner<MyOptimizer>

let optimizer = MyOptimizer(learningRate: 0.01)

// We can't schedule the learning rate because the Optimizer protocol doesn't allow setting learning rates.
// If we change it to allow setting learning rates, `ParamScheduler` should allow setting learning rates,
// with `paramKeyPath: \MyLearner.optimizer.learningRate`.
let scheduler = ParamScheduler(paramKeyPath: \MyLearner.model.sillyExtraBiasParam) { t in
    if t < 0.5 {
        return -10
    } else {
        return 0
    }
}

let recorder = Recorder<MyOptimizer>()

let learner = Learner(
    model: model,
    lossFunc: meanSquaredError,
    optimizer: optimizer,
    data: data,
    callbacks: [
        recorder,
        scheduler
    ])

learner.fit(epochs: 100)

// recorder.plotLosses()
// plt.show()


