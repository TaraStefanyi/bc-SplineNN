package network

import ActivationFunction
import Gradients
import InitializationMethod
import LAMBDA
import Outputs
import SimpleActivationFunction
import addBiasColumn
import dropLastRow
import flattenMatrixListVertically
import golem.*
import golem.matrix.Matrix
import optimizer.*
import toSingleColumn

open class StandardNN(
        protected val hiddenCounts: List<Int>,
        private val activationFunction: ActivationFunction = SimpleActivationFunction.TANH,
        private val optimizer: Optimizer = AdamOptimizer(),
        private val noiseEncoder: Double = 0.25,
        protected val layerCount: Int = hiddenCounts.size + 1
) {

    protected var inputsCount: Int = -1
    protected var outputsCount: Int = -1
    var weights: List<Matrix<Double>> = emptyList()
    var parametersCounts: List<Int> = emptyList()
    var parametersCountsCumulative: List<Int> = emptyList()
    var theta: Matrix<Double> = mat[0]

    open fun initialize(inputs: Matrix<Double>, outputs: Matrix<Double>, initMethod: InitializationMethod) {
        inputsCount = inputs.numCols()
        outputsCount = outputs.numCols()

        when (initMethod) {
            InitializationMethod.GLOROT -> {
                val rs: List<Double> = (0..hiddenCounts.size).map { layerNum ->
                    when (layerNum) {
                        hiddenCounts.size -> sqrt(6.0 / (hiddenCounts[layerNum - 1] + outputsCount))
                        0 -> sqrt(6.0 / (inputsCount + hiddenCounts[layerNum]))
                        else -> sqrt(6.0 / (hiddenCounts[layerNum - 1] + hiddenCounts[layerNum]))
                    }
                }

                //init weight with random values
                weights = (0..hiddenCounts.size).map { layerNum ->
                    when (layerNum) {
                        hiddenCounts.size -> rand(hiddenCounts[layerNum-1] + 1, outputsCount) * 2*rs[layerNum]-rs[layerNum]
                        0 -> rand(inputsCount + 1, hiddenCounts[layerNum]) * 2*rs[layerNum]-rs[layerNum]
                        else -> (rand(hiddenCounts[layerNum-1] + 1, hiddenCounts[layerNum]) * 2*rs[layerNum]-rs[layerNum])
                    }
                }



//                weights[0][0, 0 until 4] = mat[-0.59694320421237,   0.8384677463706,   0.26257962877519,  0.67684881116294]
//                weights[0][1, 0 until 4] = mat[-0.35482676613166,   0.51174161630215,  0.37532251818258, -0.79785062149002]
//                weights[1][0, 0] = -0.17722359507361
//                weights[1][1, 0] = -1.07638705599713
//                weights[1][2, 0] = 0.91992661598543
//                weights[1][3, 0] = -0.93500047977467

                // set zero to biases
                weights.forEach { it.setRow(it.numRows() - 1, fill(1, it.numCols(), 0.0)) }
                println()
            }
            InitializationMethod.AUTOENCODER -> {
                var input = inputs.copy()
                val mutableWeights = mutableListOf<Matrix<Double>>()
                for (k in 0 until hiddenCounts.size) {
                    val output = inputs.mapMat { if (Math.random() < noiseEncoder) 0.0 else it }
                    val network = StandardNN(hiddenCounts = List(1) { hiddenCounts[k] })
                    network.initialize(input, output, InitializationMethod.GLOROT)
                    network.train(input, output, 300)
                    mutableWeights.add(network.weights[0])
                    input = network.passForward(input).postActFun[0]
                    print("")
                }
                val r = sqrt(6.0 / (hiddenCounts.last() + outputsCount))
                mutableWeights.add(rand(hiddenCounts.last() + 1, outputsCount) * 2 * r - r)
                weights = mutableWeights.toList()
            }
            InitializationMethod.CUSTOM -> {
                // do nothing here
            }
        }
        theta = flattenMatrixListVertically(weights)
        parametersCounts = weights.map { it.count() }
        parametersCountsCumulative = (1..parametersCounts.size).map { parametersCounts.asSequence().take(it).sum() }
    }

    fun train(inputs: Matrix<Double>, outputs: Matrix<Double>, epochs: Int): List<Double> =
        optimizer.optimize(this, theta, inputs, outputs, epochs)

    fun test(inputs: Matrix<Double>): Matrix<Double> {
        return passForward(inputs).postActFun.last()
    }

    open fun passForward(inputs: Matrix<Double>): Outputs {
        val preActFun = mutableListOf<Matrix<Double>>()
        val postActFun = mutableListOf<Matrix<Double>>()

        //1st ITERATION: input-to-hidden pass
        //hidden-to-hidden pass
        //LAST ITERATION: hidden-to-output pass
        (0 until layerCount).forEach { hiddenLayer ->
            preActFun.add((if (postActFun.isEmpty()) inputs else postActFun.last()).addBiasColumn() * weights[hiddenLayer])
            postActFun.add(preActFun.last().mapMat { activationFunction.invoke(it) })
//            postActFun.add(if (postActFun.isEmpty()) preActFun.last() else preActFun.last().mapMat { activationFunction.invoke(it) })
        }

        return Outputs(preActFun, postActFun)
    }

    open fun passBackward(inputs: Matrix<Double>, expectedOutputs: Matrix<Double>, computedOutputs: Outputs): Gradients {
        var gradsError = zeros(theta.numRows(), 1)
        val error = expectedOutputs - computedOutputs.postActFun.last()

        //gradient for output-hidden
        var delta = -2 * error emul computedOutputs.preActFun.last().mapMat { activationFunction.derivative(it) }
        gradsError[(parametersCountsCumulative[parametersCountsCumulative.lastIndex - 1] until parametersCountsCumulative.last()), 0] = (computedOutputs.postActFun[computedOutputs.postActFun.lastIndex - 1].addBiasColumn().T * delta).toSingleColumn()

        //gradient for hidden-hidden
        (hiddenCounts.size - 1 downTo 1).forEach { hiddenLayerIndex ->
            delta = (delta * weights[hiddenLayerIndex + 1].dropLastRow().T) emul computedOutputs.preActFun[hiddenLayerIndex].mapMat { activationFunction.derivative(it) }
            gradsError[(parametersCountsCumulative[hiddenLayerIndex - 1] until parametersCountsCumulative[hiddenLayerIndex]), 0] = (computedOutputs.postActFun[hiddenLayerIndex - 1].addBiasColumn().T * delta).toSingleColumn()
        }

        //gradient for hidden-input
        delta = (delta * weights[1].dropLastRow().T) emul computedOutputs.preActFun[0].mapMat { activationFunction.derivative(it) }
        gradsError[0 until parametersCountsCumulative[0], 0] = (inputs.addBiasColumn().T * delta).toSingleColumn()

        gradsError /= expectedOutputs.numRows()
        return Gradients(gradsError, (2 * LAMBDA) * theta)
    }

    open fun reshapeWeightFromVector(theta: Matrix<Double>) {
        this.theta = theta
        var i = 0
        weights.forEach { matrix ->
            (0 until matrix.numCols()).forEach {
                matrix.setCol(it, theta[i until i + matrix.numRows(), 0])
                i += matrix.numRows()
            }
        }
    }

    open fun computeObjectiveFunction(expectedOutputs: Matrix<Double>, computedOutputs: Outputs): Double {
        return ((expectedOutputs - computedOutputs.postActFun.last()) epow 2).elementSum() / expectedOutputs.numRows() + LAMBDA * (theta epow 2).elementSum()

    }
}

fun StandardNN.test(input: List<DoubleArray>, output: List<DoubleArray>, networkIndex: Int, tolerance: Double, debug: Boolean, considerNegativeAsZero: Boolean): Boolean {
    input.forEachIndexed { index, row ->
        val result = if (considerNegativeAsZero) this.test(create(row)).mapMat { Math.max(it, 0.0) } else this.test(create(row))
        val expected = create(output[index])
        if (abs(result - expected).any { it > tolerance }) {
            if (debug) println("network: $networkIndex\t\texpected: $expected\t\tgot: $result")
            return false
        }
    }
    return true
}

fun StandardNN.computeAverageError(input: List<DoubleArray>, output: List<DoubleArray>): Double =
        input.mapIndexed { index, row ->
            val result = this.test(create(row))
            val expected = create(output[index])
            abs(result - expected).elementSum() / result.numCols()
        }.sum() / input.size

//fun network.main(args: Array<String>) {
//    val output = mat[1.0, 1.0, 1.0]
//    val input = mat[2, 2, 2]
//    val standardNN = network.StandardNN(outputsCount = 3, inputsCount = 3, hiddenCounts = List(1) {2})
//    standardNN.initialize(input, output, standardNN.initMethod)
//}