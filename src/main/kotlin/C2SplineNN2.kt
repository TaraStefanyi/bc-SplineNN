import golem.create
import golem.end
import golem.mapCols
import golem.mat
import golem.matrix.Matrix
import kotlin.math.roundToInt

val BASE_MATRIX =
        mat[
                 2,-2, 1, 1 end
                -3, 3,-2,-1 end
                 0, 0, 1, 0 end
                 1, 0, 0, 0
        ]

class C2SplineNN2(
       hiddenCounts: List<Int>,
       splineInitFunction: ActivationFunction = SimpleActivationFunction.TANH

) : SplineNN(hiddenCounts = hiddenCounts, baseMatrix = BASE_MATRIX, splineInitFunction = splineInitFunction) {

    private var derivates: List<Matrix<Double>> = emptyList()

    override fun initialize(inputs: Matrix<Double>, outputs: Matrix<Double>, initMethod: InitializationMethod) {
        super.initialize(inputs, outputs, initMethod)
        recalculateDerivates()
    }

    override fun reshapeWeightFromVector(theta: Matrix<Double>) {
        super.reshapeWeightFromVector(theta)
        recalculateDerivates()
    }

    override fun computeQMat(uIndex: Matrix<Double>, layer: Int, inputs: Int): Matrix<Double> = when (update) {
        SplineUpdate.UPDATE_ALL -> {
            create(arrayOf(
                    uIndex.mapIndexed { index, d -> values[layer][Math.max(d.roundToInt() - 1, 0), index / inputs] }.toDoubleArray(),
                    uIndex.mapIndexed { index, d -> values[layer][Math.max(d.roundToInt(), 0), index / inputs] }.toDoubleArray(),
                    uIndex.mapIndexed { index, d -> derivates[layer][Math.max(d.roundToInt() - 1, 0), index / inputs] }.toDoubleArray(),
                    uIndex.mapIndexed { index, d -> derivates[layer][Math.max(d.roundToInt(), 0), index / inputs] }.toDoubleArray()
            )).T
        }
        SplineUpdate.UPDATE_LAYER -> {
            create(arrayOf(
                    uIndex.map { d -> values[layer][Math.max(d.roundToInt() - 1, 0)] }.toDoubleArray(),
                    uIndex.map { d -> values[layer][Math.max(d.roundToInt(), 0)] }.toDoubleArray(),
                    uIndex.map { d -> derivates[layer][Math.max(d.roundToInt() - 1, 0)] }.toDoubleArray(),
                    uIndex.map { d -> derivates[layer][Math.max(d.roundToInt(), 0)] }.toDoubleArray()
            )).T
        }
        SplineUpdate.UPDATE_SINGLE -> {
            create(arrayOf(
                    uIndex.map { d -> values[0][Math.max(d.roundToInt() - 1, 0)] }.toDoubleArray(),
                    uIndex.map { d -> values[0][Math.max(d.roundToInt(), 0)] }.toDoubleArray(),
                    uIndex.map { d -> derivates[0][Math.max(d.roundToInt() - 1, 0)] }.toDoubleArray(),
                    uIndex.map { d -> derivates[0][Math.max(d.roundToInt(), 0)] }.toDoubleArray()
            )).T
        }
    }

    private fun recalculateDerivates() {
        this.derivates = this.values.map { splines -> splines.mapCols { deBoor(it) } }
    }

    private fun deBoor(splineValues: Matrix<Double>): Matrix<Double> {
        val first = splineInitFunction.derivative(controlPoints.first())
        val last = splineInitFunction.derivative(controlPoints.last())
        val n = splineValues.count()

        val rhs = DoubleArray(n - 2) { 3/samplingStep * (splineValues[it+2] - splineValues[it]) }
        rhs[0] -= first
        rhs[n - 3] -= last
        val buffer = DoubleArray(n - 2) { 0.0 }
        solveTridiagonalSystem( 1.0 , 4.0 , 1.0 , rhs , n - 2 , buffer)
        return create(arrayOf(first).toDoubleArray() + rhs + last).T
    }

    private fun solveTridiagonalSystem(lowerDiagonalValue: Double,
                                       mainDiagonalValue: Double,
                                       upperDiagonalValue: Double,
                                       rightSide: DoubleArray,
                                       numEquations: Int,
                                       buffer: DoubleArray,
                                       lastMainDiagonalValue: Double = mainDiagonalValue) {
        val m0 = 1 / mainDiagonalValue
        buffer[0] = upperDiagonalValue * m0
        rightSide[0] = rightSide[0] * m0
        val lastIndex = numEquations - 1

        for (i in 1 until lastIndex) {
            val m = 1 / (mainDiagonalValue - lowerDiagonalValue * buffer[i - 1])
            buffer[i] = upperDiagonalValue * m
            rightSide[i] = (rightSide[i] - lowerDiagonalValue * rightSide[i - 1]) * m
        }

        val mL = 1 / (lastMainDiagonalValue - lowerDiagonalValue * buffer[lastIndex - 1])
        buffer[lastIndex] = upperDiagonalValue * mL
        rightSide[lastIndex] = (rightSide[lastIndex] - lowerDiagonalValue * rightSide
                [lastIndex - 1]) * mL

        var i = numEquations - 1
        while (i-- > 0) {
            rightSide[i] = rightSide[i] - buffer[i] * rightSide[i+ 1]
        }
    }
}