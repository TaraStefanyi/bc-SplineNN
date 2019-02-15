import golem.create
import golem.mat
import java.io.File

fun main(args: Array<String>) {
    val lines = File("test_data1.txt").readLines()
    val topology = lines.first().split(" ").map { it.toInt() }
    val hiddenCounts = topology.subList(1, topology.lastIndex)
    val inputs = create(lines.asSequence().filter { it.startsWith("in: ") }.map { input -> input.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
    val outputs = create(lines.asSequence().filter { it.startsWith("out: ") }.map { output -> output.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
    val network = C2SplineNNBaseMatrix(hiddenCounts = hiddenCounts)
    val matt = network.baseMatrixForAB(0.7, 0.8)
//    val network = StandardNN(hiddenCounts = hiddenCounts)
    network.initialize(inputs, outputs, InitializationMethod.GLOROT)
    network.train(inputs, outputs)
    val output2 = network.test(mat[0, 0])
    val output3 = network.test(mat[0, 1])
    val output4 = network.test(mat[1, 0])
    val output5 = network.test(mat[1, 1])
    println("${output2[0]} expecting 0")
    println("${output3[0]} expecting 1")
    println("${output4[0]} expecting 1")
    println("${output5[0]} expecting 0")

//    var n: StandardNN
//    val file = File("results.txt")
//    (0 until 200 ).forEach { iteration ->
//        n = SplineNN(hiddenCounts = hiddenCounts)
////        n = StandardNN(hiddenCounts = hiddenCounts)
//        n.initialize(inputs, outputs, InitializationMethod.GLOROT)
//        n.weights.forEach { file.appendText("${it.repr()}\n") }
//        n.train(inputs, outputs)
//        file.appendText("test [0, 0]: got ${n.test(mat[0, 0])[0]} - expected 0\n")
//        file.appendText("test [0, 1]: got ${n.test(mat[0, 1])[0]} - expected 1\n")
//        file.appendText("test [1, 0]: got ${n.test(mat[1, 0])[0]} - expected 1\n")
//        file.appendText("test [1, 1]: got ${n.test(mat[1, 1])[0]} - expected 0\n\n\n")
//        println("finished iteration ${iteration + 1}")
//    }
}


//fun main(args: Array<String>) {
//    val lines = File("chemical.txt").readLines()
//    val topology = lines.first().substring(1).split(" ").map { it.toInt() }
//    val hiddenCounts = topology.subList(1, topology.lastIndex)
//    val inputs = create(lines.asSequence().filter { it.startsWith("in: ") }.map { input -> input.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
//    val outputs = create(lines.asSequence().filter { it.startsWith("out: ") }.map { output -> output.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList().toTypedArray())
//    val network = SplineNN(hiddenCounts = hiddenCounts)
////    val network = StandardNN(hiddenCounts = hiddenCounts)
//    network.initialize(inputs, outputs, InitializationMethod.GLOROT)
//    network.train(inputs, outputs)
//    val output2 = network.test(mat[157.00,  9596.00,   4714.00,  376.00,  2.58,  407.00,  564.00,  510354.00])
//    val output3 = network.test(mat[155.00,  9487.00,   5049.00,  381.00,  2.28,  411.00,  567.00,  504718.00])
//    val output4 = network.test(mat[154.00,  9551.00,   5070.00,  374.00,  2.98,  406.00,  563.00,  456972.00])
//    val output5 = network.test(mat[154.00,  9637.00,   5087.00,  382.00,  2.57,  408.00,  565.00,  512311.00])
//    println("${output2[0]} expecting 514.00")
//    println("${output3[0]} expecting 516.00")
//    println("${output4[0]} expecting 512.00")
//    println("${output5[0]} expecting 516.00")
//}