package network

import ActivationFunction
import Gradients
import InitializationMethod
import LAMBDA
import LAMBDA_Q0
import Outputs
import SimpleActivationFunction
import addBiasColumn
import copyColumnVectorHorizontally
import dropLastRow
import flattenMatrixListVertically
import golem.*
import golem.matrix.Matrix
import toSingleColumn
import kotlin.math.roundToInt

open class SplineNN(
        hiddenCounts: List<Int>,
        protected val splineInitFunction: ActivationFunction = SimpleActivationFunction.TANH,
        private val splineInitProbability: Double = 0.1,
        private val splineInitNoise: Double = 0.0,
        protected val samplingStep: Double = 0.05,
        private val limit: Double = 2.0,
        private val baseMatrix: Matrix<Double> = SplineType.BSPLINE.baseMatrix,
        protected val update: SplineUpdate = SplineUpdate.UPDATE_LAYER
): StandardNN(hiddenCounts = hiddenCounts) {

    protected var controlPoints: Matrix<Double> = mat[0]
    protected val values: MutableList<Matrix<Double>> = mutableListOf()
    private var initialValues: Matrix<Double> = mat[0]
    private var passForwardInfo: Triple<List<Matrix<Double>>, List<Matrix<Double>>, List<Matrix<Double>>> = Triple(emptyList(), emptyList(), emptyList())

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

        var grads = -2 * copyColumnVectorHorizontally(error.T.toList().toDoubleArray(), 4).T emul (baseMatrix.T * uVector)
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

            delta = (delta * weights[hiddenLayerIndex + 1].dropLastRow().T)
                grads = copyColumnVectorHorizontally(delta.toSingleColumn().toList().toDoubleArray(), 4).T emul (baseMatrix.T * uVector)
            gradsQ[hiddenLayerIndex] = reshapeSplineGradients(hiddenCounts[hiddenLayerIndex] * inputs.numRows(), grads, uIndex)

            delta = delta emul getSplineDerivates(computedOutputs.preActFun[hiddenLayerIndex], u, uIndex, hiddenLayerIndex).first
            gradsError[(parametersCountsCumulative[hiddenLayerIndex-1] until parametersCountsCumulative[hiddenLayerIndex]), 0] = (computedOutputs.postActFun[hiddenLayerIndex-1].addBiasColumn().T * delta).toSingleColumn()
        }

        u = passForwardInfo.first.first()
        uIndex = passForwardInfo.second.first()
        uVector = passForwardInfo.third.first().T

        delta = (delta * weights[1].dropLastRow().T)
        grads = copyColumnVectorHorizontally(delta.toSingleColumn().toList().toDoubleArray(), 4).T emul (baseMatrix.T * uVector)

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
            val range = if (intIndex == 0) 0..3 else if (intIndex > g.numRows() - 3) g.numRows()-4 until g.numRows() else intIndex-1..intIndex+2
            g[ range, it ] = grads.getCol(it)
        }
        return g
    }


    open fun getSplineValues(preActFun: Matrix<Double>, layer: Int): SplineValues {
        val indexes = getSplineIndexes(preActFun)
        val u = indexes.first.toSingleColumn()
        val uIndex = indexes.second.toSingleColumn()
        val uVector = ones(u.numRows(), 4).mapMatIndexed { row, col, _ ->  pow(u[row, 0], 3 - col) }
        val qMat = computeQMat(uIndex, layer, preActFun.numRows())

        val v = if (u.count() > 1) {
            val l = mutableListOf<Double>()
            (uVector * baseMatrix).eachRow { l.add(dot(it, qMat.getRow(l.size))) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T
        } else {
            uVector * baseMatrix * qMat.T
        }

        return SplineValues(v, u, uIndex, uVector)
    }



    protected fun getSplineIndexes(preActFun: Matrix<Double>, notInLast: Int = 4): Pair<Matrix<Double>, Matrix<Double>> {
        val su = preActFun / samplingStep + (controlPoints.numRows() - 1) / 2
        var uIndex = floor(su + 0.00001)
        val u = su - uIndex
        uIndex = uIndex.mapMat {
            when {
                it < 0 -> 0.0
                it >= controlPoints.numRows() - 1 -> (controlPoints.numRows() - 2).toDouble()
                else -> it
            }
        }

        return Pair(u, uIndex)
    }

    protected open fun getSplineDerivates(preActFun: Matrix<Double>, u: Matrix<Double>, uIndex: Matrix<Double>, layer: Int): Pair<Matrix<Double>, Matrix<Double>> {
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

        val dx = if (u.count() > 1) {
            val l = mutableListOf<Double>()
            (du * baseMatrix).eachRow { l.add(dot(it, qMat.getRow(l.size))) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T / samplingStep
        } else {
            du * baseMatrix * qMat.T / samplingStep
        }

        val ddx = if (u.count() > 1) {
            val l = mutableListOf<Double>()
            (ddu * baseMatrix).eachRow { l.add(dot(it, qMat.getRow(l.size))) }
            create(l.asSequence().chunked(preActFun.numRows()).map { it.toDoubleArray() }.toList().toTypedArray()).T / pow(samplingStep, 2)
        } else {
            ddu * baseMatrix * qMat.T / pow(samplingStep, 2)
        }

        return Pair(dx, ddx)
    }

    open fun computeQMat(uIndex: Matrix<Double>, layer: Int, inputs: Int): Matrix<Double> {
//        val qmat = create(Array(4) { i -> uIndex.mapIndexed { index, d ->
//            val row = Math.max(d.roundToInt() + i - 1, 0)
//            val col = index / inputs
//            values[layer][row, col]
//        }.toDoubleArray() }).T
        return when (update) {
            SplineUpdate.UPDATE_ALL -> {
//                val indices = mutableListOf<Int>()
//                uIndex.eachIndexed { _, col, d -> indices.customAdd(values[layer].sub2ind(d.roundToInt() - 1, col)) }
//                create(Array(4) { i -> indices.map { values[layer].T[it + i] }.toDoubleArray() }).T
                create(Array(4) { i -> uIndex.mapIndexed { index, d -> values[layer][Math.max(Math.min(d.roundToInt() + i - 1, values[layer].numRows() - 1), 0), index / inputs] }.toDoubleArray() }).T
            }
            SplineUpdate.UPDATE_LAYER -> {
                create(Array(4) { i -> uIndex.map { values[layer][Math.max(Math.min(it.roundToInt() + i - 1, values[layer].numRows() - 1), 0)] }.toDoubleArray() }).T
            }
            SplineUpdate.UPDATE_SINGLE -> {
                create(Array(4) { i -> uIndex.map { values[0][Math.max(Math.min(it.roundToInt() + i - 1, values[0].numRows() - 1) , 0)] }.toDoubleArray() }).T
            }
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

//        println()
    }

    override fun computeObjectiveFunction(expectedOutputs: Matrix<Double>, computedOutputs: Outputs): Double {
        var f = ((expectedOutputs - computedOutputs.postActFun.last()) epow 2).elementSum() / expectedOutputs.numRows() + LAMBDA *(theta[0 until parametersCountsCumulative.last(), 0] epow 2).elementSum()
        if (LAMBDA_Q0 > 0)
            f += LAMBDA_Q0 *((theta[parametersCountsCumulative.last() until theta.numRows(), 0] - initialValues) epow 2).elementSum()
        return f
    }
}

data class SplineValues(val v: Matrix<Double>, val u: Matrix<Double>, val uIndex: Matrix<Double>, val uVector: Matrix<Double>)

enum class SplineUpdate {
    UPDATE_ALL,
    UPDATE_LAYER,
    UPDATE_SINGLE
}

enum class SplineType(val baseMatrix: Matrix<Double>) {
    CATMULROM(mat[
            -1, 3,-3, 1 end
             2,-5, 4,-1 end
            -1, 0, 1, 0 end
             0, 2, 0, 0
    ]*0.5),
    BSPLINE(mat[
            -1, 3,-3, 1 end
             3,-6, 3, 0 end
            -3, 0, 3, 0 end
             1, 4, 1, 0
    ]*(1.0/6.0))
}