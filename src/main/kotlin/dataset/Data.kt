package dataset

import golem.matrix.Matrix

data class Data(
        val hiddenCounts: List<Int>,
        val tolerance: Double,
        val considerNegativeAsZero: Boolean,
        val inputMatrix: Matrix<Double>,
        val outputMatrix: Matrix<Double>,
        val inputBuckets: List<Matrix<Double>>,
        val outputsBuckets: List<Matrix<Double>>,
        val testInputList: List<DoubleArray>,
        val testOutputList: List<DoubleArray>
)