import golem.*
import golem.matrix.Matrix
import java.util.Collections.emptyList
import kotlin.math.roundToInt

class C2SplineNNBaseMatrix(
        override val hiddenCounts: List<Int>,
        private val splineInitFunction: ActivationFunction = SimpleActivationFunction.TANH,
        private val splineInitProbability: Double = 0.1,
        private val splineInitNoise: Double = 0.0,
        private val samplingStep: Double = 1.5,
        private val limit: Double = 30.0,
        private val update: SplineUpdate = SplineUpdate.UPDATE_ALL
): StandardNN(hiddenCounts = hiddenCounts) {

    private var controlPoints: Matrix<Double> = mat[0]
    private val values: MutableList<Matrix<Double>> = mutableListOf()
    private var initialValues: Matrix<Double> = mat[0]
    private var passForwardInfo: Triple<List<Matrix<Double>>, List<Matrix<Double>>, List<Matrix<Double>>> = Triple(emptyList(), emptyList(), emptyList())
    private var derivates: List<Matrix<Double>> = emptyList()

    override fun initialize(inputs: Matrix<Double>, outputs: Matrix<Double>, initMethod: InitializationMethod) {
        var i = -limit - samplingStep
        val cpList = mutableListOf<Double>()
        while (i <= limit + samplingStep + 0.0001) {
            cpList.add(i)
            i += samplingStep
        }
        controlPoints = create(cpList.toDoubleArray()).T

        super.initialize(inputs, outputs, initMethod)

        val afValues = cpList.map { splineInitFunction.invoke(it) }.toDoubleArray()

        val initialValuesList = mutableListOf<Matrix<Double>>()
        when (update) {
            SplineUpdate.UPDATE_ALL -> {
                hiddenCounts.forEach { initialValuesList.add(copyColumnVectorHorizontally(afValues, it)) }
                initialValuesList.add(copyColumnVectorHorizontally(afValues, outputsCount))
            }
            SplineUpdate.UPDATE_LAYER -> {
                (0 until layerCount).forEach { initialValuesList.add(copyColumnVectorHorizontally(afValues, 1)) }
            }
            SplineUpdate.UPDATE_SINGLE -> {
                initialValuesList.clear()
                initialValuesList.add(copyColumnVectorHorizontally(afValues, 1))
            }
        }
        values.addAll(initialValuesList)

        values.forEach { matrix ->
            List(matrix.count()) {it}
                    .shuffled()
                    .take(ceil(splineInitProbability * matrix.count()))
                    .forEach { matrix[it] += randn(1)[0] * splineInitNoise }
        }

        initialValues = flattenMatrixListVertically(initialValuesList)
        theta = create(theta.T.to2DArray()[0] + flattenMatrixListVertically(values).T.to2DArray()[0]).T
    }

    override fun passForward(inputs: Matrix<Double>): Outputs {
        val preActFun = mutableListOf<Matrix<Double>>()
        val postActFun = mutableListOf<Matrix<Double>>()
        val u = mutableListOf<Matrix<Double>>()
        val uIndex = mutableListOf<Matrix<Double>>()
        val uVector = mutableListOf<Matrix<Double>>()

        //1st ITERATION: input-to-hidden pass
        //hidden-to-hidden pass
        //LAST ITERATION: hidden-to-output pass
        (0 until layerCount).forEach { hiddenLayer ->
            preActFun.add((if (postActFun.isEmpty()) inputs else postActFun.last()).addBiasColumn()*weights[hiddenLayer])
            val splineValues = getSplineValues(preActFun.last(), hiddenLayer)
            postActFun.add(splineValues.v)
            u.add(splineValues.u)
            uIndex.add(splineValues.uIndex)
            uVector.add(splineValues.uVector)
        }

        passForwardInfo = Triple(u, uIndex, uVector)
        return Outputs(preActFun, postActFun)
    }

    override fun passBackward(inputs: Matrix<Double>, expectedOutputs: Matrix<Double>, computedOutputs: Outputs): Gradients {
        var gradsError = zeros(theta.numRows(), 1)
        val gradsQ = MutableList(layerCount) { mat[0] }
        val error = expectedOutputs - computedOutputs.postActFun.last()
        // splajn definovany  po castiach, v uindex je spodna hranica (kontrolny bod - jeho index)  - teda urcuje index intervalu
        //v u je xova suradnica na konkretnom intervale
        //uvector je u 4*za sebou u3, u2, u, 1
        var u = passForwardInfo.first.last()
        var uIndex = passForwardInfo.second.last()
        var uVector = passForwardInfo.third.last().T
        var baseMatrixes = baseMatrixes(uIndex, false)

        var grads = -2 * copyColumnVectorHorizontally(error.T.toList().toDoubleArray(), 4).T emul uVector.mapColsIndexed { i, col -> baseMatrixes[i].T * col}
        gradsQ[gradsQ.lastIndex] = reshapeSplineGradients(outputsCount * inputs.numRows(), grads, uIndex)
        var delta = -2 * error emul getSplineDerivates(computedOutputs.preActFun.last(), u, uIndex, layerCount - 1).first
        //grads - tu sa vlozi ta delta (chyba) * uvector*B - to sa umiesti podla uindex
        //gradsq - matcia - v riadky - pocet kontrolnych bodov, stplce su jednotlive neurony
        //
        gradsError[(parametersCountsCumulative[parametersCountsCumulative.lastIndex-1] until parametersCountsCumulative.last()), 0] = (computedOutputs.postActFun[computedOutputs.postActFun.lastIndex-1].addBiasColumn().T * delta).toSingleColumn()

        (hiddenCounts.lastIndex downTo 1).forEach { hiddenLayerIndex ->
            u = passForwardInfo.first[hiddenLayerIndex]
            uIndex = passForwardInfo.second[hiddenLayerIndex]
            uVector = passForwardInfo.third[hiddenLayerIndex].T
            baseMatrixes = baseMatrixes(uIndex, false)

            delta = (delta * weights[hiddenLayerIndex + 1].dropLastRow().T)
            grads = copyColumnVectorHorizontally(delta.toSingleColumn().toList().toDoubleArray(), 4).T emul uVector.mapColsIndexed { i, col -> baseMatrixes[i].T * col}
            gradsQ[hiddenLayerIndex] = reshapeSplineGradients(hiddenCounts[hiddenLayerIndex] * inputs.numRows(), grads, uIndex)

            delta = delta emul getSplineDerivates(computedOutputs.preActFun[hiddenLayerIndex], u, uIndex, hiddenLayerIndex).first
            gradsError[(parametersCountsCumulative[hiddenLayerIndex-1] until parametersCountsCumulative[hiddenLayerIndex]), 0] = (computedOutputs.postActFun[hiddenLayerIndex-1].addBiasColumn().T * delta).toSingleColumn()
        }

        u = passForwardInfo.first.first()
        uIndex = passForwardInfo.second.first()
        uVector = passForwardInfo.third.first().T
        baseMatrixes = baseMatrixes(uIndex, false)

        delta = (delta * weights[1].dropLastRow().T)
        grads = copyColumnVectorHorizontally(delta.toSingleColumn().toList().toDoubleArray(), 4).T emul uVector.mapColsIndexed { i, col -> baseMatrixes[i].T * col}

        gradsQ[0] = reshapeSplineGradients(hiddenCounts[0] * inputs.numRows(), grads, uIndex)

        when (update) {

            SplineUpdate.UPDATE_ALL -> {
                var index = parametersCountsCumulative.last()
                (0..hiddenCounts.size).forEach {layer ->
                    val neurons = if (layer == hiddenCounts.size) outputsCount else hiddenCounts[layer]
                    gradsError[index until index + neurons*controlPoints.numRows(), 0] = create(Array(neurons) { neuron ->
                        DoubleArray(controlPoints.numRows()) {controlPoint ->
                            gradsQ[layer][controlPoint, neuron*inputs.numRows() until (neuron+1)*inputs.numRows()].elementSum()
                        }
                    }).T.toSingleColumn()
                    index += neurons*controlPoints.numRows()
                }
            }
            SplineUpdate.UPDATE_LAYER -> {
                var index = parametersCountsCumulative.last()
                (0..hiddenCounts.size).forEach { layer ->
                    gradsError[index until index + controlPoints.numRows(), 0] = create(DoubleArray(controlPoints.numRows()) {controlPoint ->
                        gradsQ[layer].getRow(controlPoint).elementSum()
                    }).T
                    index += controlPoints.numRows()
                }
            }
            SplineUpdate.UPDATE_SINGLE -> {
                (0..hiddenCounts.size).forEach { layer ->
                    val index = parametersCountsCumulative.last()
                    gradsError[index until index + controlPoints.numRows(), 0] += create(DoubleArray(controlPoints.numRows()) {controlPoint ->
                        gradsQ[layer].getRow(controlPoint).elementSum()
                    }).T
                }
            }
        }

        delta = delta emul getSplineDerivates(computedOutputs.preActFun.first(), u, uIndex, 0).first
        gradsError[0 until parametersCountsCumulative.first(), 0] = (inputs.addBiasColumn().T * delta).toSingleColumn()

        gradsError /= expectedOutputs.numRows()
        val gradsWeights = create(
                DoubleArray(parametersCountsCumulative.last()) {
                    2 * LAMBDA * theta[it]
                } + DoubleArray(theta.numRows() - parametersCountsCumulative.last()) {
                    2 * LAMBDA_Q0 * (theta[it + parametersCountsCumulative.last()] - initialValues[it])
                }
        ).T
        return Gradients(gradsError, gradsWeights)
    }

    private fun reshapeSplineGradients(s: Int, grads: Matrix<Double>, uIndex: Matrix<Double>): Matrix<Double> {
        val g = zeros(controlPoints.numRows(), s)
        (0 until s).forEach {
            val intIndex = uIndex[it].roundToInt()
            val range = if (intIndex == 0) 0..3 else intIndex-1..intIndex+2
            g[ range, it ] = grads.getCol(it)
        }
        return g
    }


    private fun getSplineValues(preActFun: Matrix<Double>, layer: Int): SplineValues {
        val indexes = getSplineIndexes(preActFun)
        val u = indexes.first.toSingleColumn()
        val uIndex = indexes.second.toSingleColumn()
        val uVector = ones(u.numRows(), 4).mapMatIndexed { row, col, _ ->  pow(u[row, 0], 3 - col) }
        val qMat = computeQMat(uIndex, layer, preActFun.numRows())

        val baseMatrixes = baseMatrixes(uIndex)
//        val baseMatrixes= listOf(SplineType.CATMULROM.baseMatrix)
        val v = if (u.count() > 1) {
//            val l = uVector.mapRowsToListIndexed { i, row -> dot(row * baseMatrixes[i], qMat.getRow(i)) }
            val l = mutableListOf<Double>()
            (uVector * baseMatrixes[0]).eachRow { l.add(dot(it, qMat.getRow(l.size))) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T
        } else {
            uVector * baseMatrixes.first() * qMat.T
        }

        return SplineValues(v, u, uIndex, uVector)
    }



    private fun getSplineIndexes(preActFun: Matrix<Double>, notInLast: Int = 4): Pair<Matrix<Double>, Matrix<Double>> {
        val su = preActFun / samplingStep + (controlPoints.numRows() - 1) / 2
        var uIndex = floor(su + 0.00001)
        val u = su - uIndex
        uIndex = uIndex.mapMat {
            when {
                it < 1 -> 1.0
                it > controlPoints.numRows() - notInLast -> (controlPoints.numRows() - notInLast).toDouble()
                else -> it
            }
        }

        return Pair(u, uIndex)
    }

    private fun getSplineDerivates(preActFun: Matrix<Double>, u: Matrix<Double>, uIndex: Matrix<Double>, layer: Int): Pair<Matrix<Double>, Matrix<Double>> {
        val du = create(arrayOf(
                (3 * (u epow 2)).T.to2DArray()[0],
                (2 * u).T.to2DArray()[0],
                DoubleArray(u.numRows()) {1.0},
                DoubleArray(u.numRows()) {0.0}
        )).T

        val ddu = create(arrayOf(
                (6 * u).T.to2DArray()[0],
                DoubleArray(u.numRows()) {1.0},
                DoubleArray(u.numRows()) {0.0},
                DoubleArray(u.numRows()) {0.0}
        )).T

        val qMat = computeQMat(uIndex, layer, preActFun.numRows())
        val baseMatrixes = baseMatrixes(uIndex, true)

        val dx = if (u.count() > 1) {
            val l = du.mapRowsToListIndexed { i, row -> dot(row * baseMatrixes[i], qMat.getRow(i)) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T / samplingStep
        } else {
            du * baseMatrixes.first() * qMat.T / samplingStep
        }

        val ddx = if (u.count() > 1) {
            val l = ddu.mapRowsToListIndexed { i, row -> dot(row * baseMatrixes[i], qMat.getRow(i)) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T / pow(samplingStep, 2)
        } else {
            ddu * baseMatrixes.first() * qMat.T / pow(samplingStep, 2)
        }

        return Pair(dx, ddx)
    }

    private fun computeQMat(uIndex: Matrix<Double>, layer: Int, inputs: Int): Matrix<Double> = when (update) {
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

    override fun reshapeWeightFromVector(theta: Matrix<Double>) {
        super.reshapeWeightFromVector(theta[0 until parametersCountsCumulative.last(), 0])
        this.theta = theta
        var i = parametersCountsCumulative.last()
        values.forEach { matrix ->
            (0 until matrix.numCols()).forEach {
                matrix.setCol(it, theta[i until i + matrix.numRows(), 0])
                i += matrix.numRows()
            }
        }
        recalculateDerivates()
    }

    override fun computeObjectiveFunction(expectedOutputs: Matrix<Double>, computedOutputs: Outputs): Double {
        var f = ((expectedOutputs - computedOutputs.postActFun.last()) epow 2).elementSum() / expectedOutputs.numRows() + LAMBDA*(theta[0 until parametersCountsCumulative.last(), 0] epow 2).elementSum()
        if (LAMBDA_Q0 > 0)
            f += LAMBDA_Q0*((theta[parametersCountsCumulative.last() until theta.numRows(), 0] - initialValues) epow 2).elementSum()
        return f
    }

    private fun baseMatrixes(uIndex: Matrix<Double>, derivate: Boolean = false) = uIndex.mapRowsToList {
        if (derivate)
            baseMatrixDerivativeForAB(controlPoints[it[0].roundToInt()], controlPoints[it[0].roundToInt() + 1])
        else
            baseMatrixForAB(controlPoints[it[0].roundToInt()], controlPoints[it[0].roundToInt() + 1])
    }

//    fun baseMatrixForAB(a: Double, b: Double) = mat[
//            (3*(a pow 3) - (a pow 2)*b) / ((a - b) pow 3),
//            (a*(b pow 2) - 3*(b pow 3)) / ((a - b) pow 3),
//            (-(a pow 2)*(b pow 2) + a*(b pow 3)) / ((a - b) pow 3),
//            (-(a pow 3)*b + (a pow 2)*(b pow 2)) / ((a - b) pow 3)
//        end
//            (- 8*(a pow 2) + 2*a*b) / ((a - b) pow 3),
//            (-2*a*b + 8*(b pow 2)) / ((a - b) pow 3),
//            (2*(a pow 2)*b - a*(b pow 2) - (b pow 3)) / ((a - b) pow 3),
//            ((a pow 3) + (a pow 2)*b  - 2*a*(b pow 2)) / ((a - b) pow 3)
//        end
//            (7*a - b) / ((a - b) pow 3),
//            (a - 7*b) / ((a - b) pow 3),
//            (-(a pow 2) - a*b + 2*(b pow 2)) / ((a - b) pow 3),
//            (-2*(a pow 2) + a*b + (b pow 2)) / ((a - b) pow 3)
//        end
//            -2 / ((a - b) pow 3),
//            2 / ((a - b) pow 3),
//            1 / ((a - b) pow 2),
//            1 / ((a - b) pow 2)
//    ]

    fun baseMatrixForAB(a: Double, b: Double) = mat[
            (-(a pow 3)*b + (a pow 2)*(b pow 2)) / ((a - b) pow 3),
            (-(a pow 2)*(b pow 2) + a*(b pow 3)) / ((a - b) pow 3),
            (a*(b pow 2) - 3*(b pow 3)) / ((a - b) pow 3),
            (3*(a pow 3) - (a pow 2)*b) / ((a - b) pow 3)
         end
            ((a pow 3) + (a pow 2)*b  - 2*a*(b pow 2)) / ((a - b) pow 3),
            (2*(a pow 2)*b - a*(b pow 2) - (b pow 3)) / ((a - b) pow 3),
            (-2*a*b + 8*(b pow 2)) / ((a - b) pow 3),
            (- 8*(a pow 2) + 2*a*b) / ((a - b) pow 3)
         end
            (-2*(a pow 2) + a*b + (b pow 2)) / ((a - b) pow 3),
            (-(a pow 2) - a*b + 2*(b pow 2)) / ((a - b) pow 3),
            (a - 7*b) / ((a - b) pow 3),
            (7*a - b) / ((a - b) pow 3)
         end
            1 / ((a - b) pow 2),
            1 / ((a - b) pow 2),
            2 / ((a - b) pow 3),
            -2 / ((a - b) pow 3)
    ]

//    fun baseMatrixDerivativeForAB(a: Double, b: Double) = mat[
//            (-6*a*b) / ((a - b) pow 3),
//            (6*a*b) / ((a - b) pow 3),
//            (2*(a pow 2)*b - a*(b pow 2) - (b pow 3)) / ((a - b) pow 3),
//            ((a pow 3) + (a pow 2)*b - 2*a*(b pow 2)) / ((a - b) pow 3)
//        end
//            (6*a + 6*b) / ((a - b) pow 3),
//            (-6*a - 6*b) / ((a - b) pow 3),
//            (-2*(a pow 2) - 2*a*b + 4*(b pow 2)) / ((a - b) pow 3),
//            (-4*(a pow 2) + 2*a*b + 2*(b pow 2)) / ((a - b) pow 3)
//        end
//            (-6) / ((a - b) pow 3),
//            (6) / ((a - b) pow 3),
//            (3*a - 3*b) / ((a - b) pow 3),
//            (3*a - 3*b) / ((a - b) pow 3)
//        end
//            0,
//            0,
//            0,
//            0
//    ]

    fun baseMatrixDerivativeForAB(a: Double, b: Double) = mat[
            ((a pow 3) + (a pow 2)*b - 2*a*(b pow 2)) / ((a - b) pow 3),
            (2*(a pow 2)*b - a*(b pow 2) - (b pow 3)) / ((a - b) pow 3),
            (6*a*b) / ((a - b) pow 3),
            (-6*a*b) / ((a - b) pow 3)
         end
            (-4*(a pow 2) + 2*a*b + 2*(b pow 2)) / ((a - b) pow 3),
            (-2*(a pow 2) - 2*a*b + 4*(b pow 2)) / ((a - b) pow 3),
            (-6*a - 6*b) / ((a - b) pow 3),
            (6*a + 6*b) / ((a - b) pow 3)
         end
            (3*a - 3*b) / ((a - b) pow 3),
            (3*a - 3*b) / ((a - b) pow 3),
            (6) / ((a - b) pow 3),
            (-6) / ((a - b) pow 3)
         end
            0,
            0,
            0,
            0
    ]

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