import golem.*
import golem.matrix.Matrix

class AdamOptimizer: Optimizer {
    override fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>) {
        val epochs = 1000
        val batchSize = 300
        val beta1 = 0.9
        val beta2 = 0.999
        val eta = 0.001
        val eps = pow(10, -8)

        var currentIdx = 0
        var w = weights.copy()
        val i = inputs.copy()
        val o = outputs.copy()
        val fval = zeros(epochs, 1)
        val gval = zeros(epochs, 1)

        var mt = zeros(w.numRows(), 1)
        var vt = zeros(w.numRows(), 1)

        var idx: IntRange

        for (n in 0 until epochs) {
            if (currentIdx + batchSize > i.numRows()) {
                idx = currentIdx until i.numRows()
                currentIdx = 0
                if (batchSize < Double.POSITIVE_INFINITY) {
                    //shuffle??
                    shuffle(i, o)
                }
            } else {
                idx = currentIdx until currentIdx+batchSize
                currentIdx += batchSize
            }

            val inputsBatch = i[idx, 0 until i.numCols()]
            val outputsBatch = o[idx, 0 until o.numCols()]

            var computedOutputsBatch = network.passForward(inputsBatch)
            var g = network.passBackward(inputsBatch, outputsBatch, computedOutputsBatch)
            var grads = g.gradsError + g.gradsWeights

            mt = mt * beta1 +  grads * (1 - beta1)
            vt = vt * beta2 +  (grads epow 2) * (1 - beta2)
            var mtHat = mt / (1 - pow(beta1, n+1))
            var vtHat = vt / (1 - pow(beta2, n+1))

            w = w - (mtHat * eta) emul (sqrt(vtHat + eps) epow -1)

            fval[n, 0] = network.computeObjectiveFunction(outputsBatch, computedOutputsBatch)
            gval[n, 0] = grads.norm()

            network.reshapeWeightFromVector(w)

        }
    }

    private fun shuffle(inputs: Matrix<Double>, outputs: Matrix<Double>) {
        val s = List(inputs.numRows()) {it}.shuffled()
        val i = inputs.copy()
        val o = outputs.copy()
        val inputsColRange = 0 until i.numCols()
        val outputsColRange = 0 until o.numCols()
        s.forEachIndexed { newIndex, oldIndex ->
            inputs[newIndex, inputsColRange] = i.getRow(oldIndex)
            outputs[newIndex, outputsColRange] = o.getRow(oldIndex)
        }
    }

}