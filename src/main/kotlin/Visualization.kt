import golem.create
import golem.mat
import golem.plot
import golem.pow
import java.io.File

fun main() {
    val lines = File("test_data1.txt").readLines()
    val topology = lines.first().split(" ").map { it.toInt() }
    val hiddenCounts = topology.subList(1, topology.lastIndex)
    val inputs = create(lines.asSequence().filter { it.startsWith("in: ") }.map { input -> input.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
    val outputs = create(lines.asSequence().filter { it.startsWith("out: ") }.map { output -> output.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
//    val network = SplineNN(hiddenCounts = hiddenCounts, splineInitFunction = SimpleActivationFunction.RELU)
    val network = C2SplineNN2(hiddenCounts = hiddenCounts, splineInitFunction = SimpleActivationFunction.TANH)
    network.initialize(inputs, outputs, InitializationMethod.GLOROT)

    val x = DoubleArray((400).toInt()) {it / 100.0 - 2}
//    val x = arrayOf(0.0, 0.025, 0.033, 0.05, 0.066, 0.075, 0.1).toDoubleArray()
    plot(x, network.getSplineValues(create(x).T, 1).v, color = "red")

    network.train(inputs, outputs)
    plot(x, network.getSplineValues(create(x).T, 1).v)

//    println(SimpleActivationFunction.RELU.invoke(0.0))
//    println(SimpleActivationFunction.RELU.invoke(0.1))
//    println(SimpleActivationFunction.RELU.derivative(0.0))
//    println(SimpleActivationFunction.RELU.derivative(0.1))
//    plot(x, getSplineValue(x))
//    plot(x, getSplineValueC1(x), color = "#FF0000")
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
