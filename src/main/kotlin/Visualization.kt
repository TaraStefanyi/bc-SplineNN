import dataset.XorDataset
import golem.create
import golem.mat
import golem.plot
import golem.pow
import network.*

fun main() {
    val data = XorDataset().data
//    val network = network.SplineNN(hiddenCounts = hiddenCounts, splineInitFunction = SimpleActivationFunction.RELU)
    val network = SplineNN(hiddenCounts = data.hiddenCounts, splineInitFunction = SimpleActivationFunction.TANH)
    network.initialize(data.inputMatrix, data.outputMatrix, InitializationMethod.GLOROT)

    val x = DoubleArray((400).toInt()) {it / 100.0 - 2}
//    val x = arrayOf(0.0, 0.025, 0.033, 0.05, 0.066, 0.075, 0.1).toDoubleArray()
    plot(x, network.getSplineValues(create(x).T, 1).v, color = "red")

    network.train(data.inputMatrix, data.outputMatrix, 300)
    plot(x, network.getSplineValues(create(x).T, 1).v)

}

fun getSplineValue(x: DoubleArray) =
        x.map { (mat[(it pow 3), (it pow 2), it, 1] * BASE_MATRIX * mat[
                0.0,
                0.1,
                1.0,
                1.0
        ].T)[0] }.toDoubleArray()

fun getSplineValueC1(x: DoubleArray) =
        x.map { (mat[(it pow 3), (it pow 2), it, 1] * SplineType.CATMULROM.baseMatrix * mat[
                SimpleActivationFunction.TANH.invoke(-0.1),
                SimpleActivationFunction.TANH.invoke(0.0),
                SimpleActivationFunction.TANH.invoke(0.1),
                SimpleActivationFunction.TANH.invoke(0.2)
        ].T)[0] }.toDoubleArray()
