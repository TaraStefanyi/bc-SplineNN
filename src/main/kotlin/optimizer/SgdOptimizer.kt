package optimizer

import golem.matrix.Matrix
import network.StandardNN
import shuffled

class SgdOptimizer : Optimizer {
    override fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>, epochs: Int): List<Double> {
        val learningRate = 0.05
        val loss = mutableListOf<Double>()

        for (n in 0 until epochs) {
            val (i, o) = shuffled(inputs, outputs)

            val computedOutputs = network.passForward(i)
            val g = network.passBackward(i, o, computedOutputs)
            val grads = g.gradsError + g.gradsWeights
            loss.add(network.computeObjectiveFunction(o, computedOutputs))
            network.reshapeWeightFromVector(weights - (grads * learningRate))
        }
        return loss
    }
}