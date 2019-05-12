package optimizer

import golem.matrix.Matrix
import golem.zeros
import network.StandardNN
import shuffled

class SgdMomentumOptimizer : Optimizer {
    override fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>, epochs: Int): List<Double> {
        val learningRate = 0.05
        val momentumRate = 0.01
        val loss = mutableListOf<Double>()

        var update = zeros(weights.numRows(), weights.numCols())

        for (n in 0 until epochs) {
            val (i, o) = shuffled(inputs, outputs)

            val computedOutputs = network.passForward(i)
            val g = network.passBackward(i, o, computedOutputs)
            val grads = g.gradsError + g.gradsWeights
            update = update*momentumRate + grads*(1 - momentumRate)
            loss.add(network.computeObjectiveFunction(o, computedOutputs))
            network.reshapeWeightFromVector(weights - (update * learningRate))
        }
        return loss
    }
}