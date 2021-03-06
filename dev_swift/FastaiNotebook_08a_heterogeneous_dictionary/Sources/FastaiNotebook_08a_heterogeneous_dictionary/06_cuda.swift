/*
THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
file to edit: 06_cuda.ipynb

*/
        
import Path
import TensorFlow
import Python

extension Learner {
    public class AddChannel: Delegate {
        public override func batchWillStart(learner: Learner) {
            learner.currentInput = learner.currentInput!.expandingShape(at: -1)
        }
    }
    
    public func makeAddChannel() -> AddChannel { return AddChannel() }
}

extension Array: Layer where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output
    
    @differentiable(vjp: _vjpApplied)
    public func call(_ input: Input) -> Output {
        var activation = input
        for layer in self {
            activation = layer(activation)
        }
        return activation
    }
    
    public func _vjpApplied(_ input: Input)
        -> (Output, (Output.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector))
    {
        var activation = input
        var pullbacks: [(Input.CotangentVector) -> (Element.CotangentVector, Input.CotangentVector)] = []
        for layer in self {
            let (newActivation, newPullback) = layer.valueWithPullback(at: activation) { $0($1) }
            activation = newActivation
            pullbacks.append(newPullback)
        }
        func pullback(_ v: Input.CotangentVector) -> (Array.CotangentVector, Input.CotangentVector) {
            var activationGradient = v
            var layerGradients: [Element.CotangentVector] = []
            for pullback in pullbacks.reversed() {
                let (newLayerGradient, newActivationGradient) = pullback(activationGradient)
                activationGradient = newActivationGradient
                layerGradients.append(newLayerGradient)
            }
            return (Array.CotangentVector(layerGradients.reversed()), activationGradient)
        }
        return (activation, pullback)
    }
}

public func conv<Scalar>(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 2) -> FAConv2D<Scalar> {
    return FAConv2D<Scalar>(filterShape: (ks, ks, cIn, cOut), 
                            strides: (stride,stride), 
                            padding: .same, 
                            activation: relu)
}

public struct CnnModel: Layer {
    public var convs: [FAConv2D<Float>]
    public var pool = FAAdaptiveAvgPool2D<Float>()
    public var flatten = Flatten<Float>()
    public var linear: FADense<Float>
    
    public init(channelIn: Int, nOut: Int, filters: [Int]){
        convs = []
        let allFilters = [channelIn] + filters
        for i in 0..<filters.count { convs.append(conv(allFilters[i], allFilters[i+1])) }
        linear = FADense<Float>(inputSize: filters.last!, outputSize: nOut)
    }
    
    @differentiable
    public func call(_ input: TF) -> TF {
        return input.sequenced(through: convs, pool, flatten, linear)
    }
}
