import golem.matrix.Matrix

interface Optimizer {
    fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>)
}