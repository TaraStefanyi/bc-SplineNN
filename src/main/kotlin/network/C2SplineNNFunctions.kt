package network

import InitializationMethod
import golem.*
import golem.matrix.Matrix
import toSingleColumn
import kotlin.math.roundToInt

class C2SplineNNFunctions(hiddenCounts: List<Int>) : SplineNN(hiddenCounts = hiddenCounts) {

    private var derivates: List<Matrix<Double>> = emptyList()

    override fun initialize(inputs: Matrix<Double>, outputs: Matrix<Double>, initMethod: InitializationMethod) {
        super.initialize(inputs, outputs, initMethod)
        recalculateDerivates()
    }

    override fun getSplineValues(preActFun: Matrix<Double>, layer: Int): SplineValues {
        val indexes = getSplineIndexes(preActFun, 2)
        val u = indexes.first.toSingleColumn()
        val uIndex = indexes.second
        val uVector = ones(u.numRows(), 4).mapMatIndexed { row, col, _ ->  pow(u[row, 0], 3 - col) }

        val lower = uIndex.mapMat { controlPoints[it.roundToInt()] }
        val upper = uIndex.mapMat { controlPoints[it.roundToInt() + 1] }

        val y = uIndex.mapMatIndexed { _, col, n -> values[layer][n.roundToInt(), col] }
        val y1 = uIndex.mapMatIndexed { _, col, n -> values[layer][n.roundToInt() + 1, col] }

        val d = uIndex.mapMatIndexed { _, col, n -> derivates[layer][n.roundToInt(), col] }
        val d1 = uIndex.mapMatIndexed { _, col, n -> derivates[layer][n.roundToInt() + 1, col] }

        val v = (y emul h0(preActFun, lower, upper)) +
                (y1 emul h0(preActFun, upper, lower)) +
                (d emul h1(preActFun, lower, upper)) +
                (d1 emul h1(preActFun, upper, lower))
        return SplineValues(v, u, uIndex.toSingleColumn(), uVector)
    }




    override fun getSplineDerivates(preActFun: Matrix<Double>, u: Matrix<Double>, uIndex: Matrix<Double>, layer: Int): Pair<Matrix<Double>, Matrix<Double>> {
        val indexes = getSplineIndexes(preActFun, 2).second
        val lower = indexes.mapMat { controlPoints[it.roundToInt()] }
        val upper = indexes.mapMat { controlPoints[it.roundToInt() + 1] }

        val y = indexes.mapMatIndexed { _, col, n -> values[layer][n.roundToInt(), col] }
        val y1 = indexes.mapMatIndexed { _, col, n -> values[layer][n.roundToInt() + 1, col] }

        val d = indexes.mapMatIndexed { _, col, n -> derivates[layer][n.roundToInt(), col] }
        val d1 = indexes.mapMatIndexed { _, col, n -> derivates[layer][n.roundToInt() + 1, col] }

        val v = (y emul h0d(preActFun, lower, upper)) +
                (y1 emul h0d(preActFun, upper, lower)) +
                (d emul h1d(preActFun, lower, upper)) +
                (d1 emul h1d(preActFun, upper, lower))
        return Pair(v, v)
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

    companion object {
        @JvmStatic
        private fun h0(x: Matrix<Double>, a: Matrix<Double>, b: Matrix<Double>) =
                (1 - ((2 * (x - a)) emul ((a - b) epow -1))) emul (((x - a) epow 2) emul ((a - b) epow -2))

        @JvmStatic
        private fun h1(x: Matrix<Double>, a: Matrix<Double>, b: Matrix<Double>) =
                ((b - a) epow -2) emul ((x - b) epow 2) emul (x - a)

        @JvmStatic
        private fun h0d(x: Matrix<Double>, a: Matrix<Double>, b: Matrix<Double>) =
                (6 * ((b - a) epow -3)) emul (x - b) emul (x - a)

        @JvmStatic
        private fun h1d(x: Matrix<Double>, a: Matrix<Double>, b: Matrix<Double>) =
                ((b - a) epow -2) emul (x - b) emul (3*x - b - 2*a)
    }
}

fun main(args: Array<String>) {
    val a = mat[1, 2, 3, 100 end 4, 5, 6, 945 end 5, 65, 643, 32]
    val b = mat[1, 2, 3 end 3, 60, 4 end 5, 85, 6 end 7, 543, 8]
    println(a * b)
    println()
    println((b.T* a.T))
}