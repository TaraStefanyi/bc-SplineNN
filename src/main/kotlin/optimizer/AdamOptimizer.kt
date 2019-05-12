package optimizer

import network.StandardNN
import golem.*
import golem.matrix.Matrix
import shuffled

class AdamOptimizer: Optimizer {
    override fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>, epochs: Int): List<Double> {
        val batchSize = 300
        val beta1 = 0.9
        val beta2 = 0.999
        val eta = 0.001
        val eps = pow(10, -8)

        var currentIdx = 0
        var w = weights.copy()
        var i = inputs.copy()
        var o = outputs.copy()
        val fval = zeros(epochs, 1)
        val gval = zeros(epochs, 1)

        var mt = zeros(w.numRows(), 1)
        var vt = zeros(w.numRows(), 1)

        var idx: IntRange

        val loss = mutableListOf<Double>()

        for (n in 0 until epochs) {
            if (currentIdx + batchSize > i.numRows()) {
                idx = currentIdx until i.numRows()
                currentIdx = 0
                if (batchSize < Double.POSITIVE_INFINITY) {
                    //shuffle??
                    shuffled(inputs, outputs).let {
                        i = it.first
                        o = it.second
                    }
                }
            } else {
                idx = currentIdx until currentIdx+batchSize
                currentIdx += batchSize
            }

            val inputsBatch = i[idx, 0 until i.numCols()]
            val outputsBatch = o[idx, 0 until o.numCols()]

            val computedOutputsBatch = network.passForward(inputsBatch)
            val g = network.passBackward(inputsBatch, outputsBatch, computedOutputsBatch)
            val grads = g.gradsError + g.gradsWeights

            loss.add(network.computeObjectiveFunction(outputsBatch, computedOutputsBatch))

            mt = mt * beta1 +  grads * (1 - beta1)
            vt = vt * beta2 +  (grads epow 2) * (1 - beta2)
            val mtHat = mt / (1 - pow(beta1, n+1))
            val vtHat = vt / (1 - pow(beta2, n+1))

            w -= ((mtHat * eta) emul ((sqrt(vtHat) + eps) epow -1))

            fval[n, 0] = network.computeObjectiveFunction(outputsBatch, computedOutputsBatch)
            gval[n, 0] = grads.norm()

            network.reshapeWeightFromVector(w)
        }
        return loss
    }
}