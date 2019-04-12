import golem.create
import golem.end
import golem.mat
import golem.plot
import java.io.File

fun main() {
    val lines = File("test_data1.txt").readLines()
    val topology = lines.first().split(" ").map { it.toInt() }
    val hiddenCounts = topology.subList(1, topology.lastIndex)
    val inputs = create(lines.asSequence().filter { it.startsWith("in: ") }.map { input -> input.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
    val outputs = create(lines.asSequence().filter { it.startsWith("out: ") }.map { output -> output.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
//    val network = SplineNN(hiddenCounts = hiddenCounts, splineInitFunction = SimpleActivationFunction.TANH)
    val network = C2SplineNN2(hiddenCounts = hiddenCounts, splineInitFunction = SimpleActivationFunction.TANH)
    network.initialize(inputs, outputs, InitializationMethod.GLOROT)

    val x = DoubleArray((400).toInt()) {it / 100.0 - 2}
    plot(x, network.getSplineValues(create(x).T, 0).v, color = "red")

    network.train(inputs, outputs)
    plot(x, network.getSplineValues(create(x).T, 0).v)

}