import golem.abs
import golem.create
import golem.to2DArray
import java.io.File

fun main() {
    val runs = 100
    val debug = false
    val actFun = SimpleActivationFunction.SIG
    val dataset = Dataset.XOR

    val testResults = List(runs) {dataset.doTest(it, actFun, debug = debug)}

    val successfulRuns = testResults
            .map { results -> results.map { if (it.first) 1 else 0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .map { it.toDouble() }
    println("Success rate: $successfulRuns")

    val averageErrorSuccess = testResults
            .map { results -> results.map { if (it.first) it.third else 0.0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .mapIndexed { i, totalError -> totalError / successfulRuns[i] }
    println("Average error success: $averageErrorSuccess")

    val averageErrorFail = testResults
            .map { results -> results.map { if (!it.first) it.third else 0.0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .mapIndexed { i, totalError -> totalError / (runs - successfulRuns[i]) }
    println("Average error fail: $averageErrorFail")
}



fun StandardNN.test(input: List<DoubleArray>, output: List<DoubleArray>, networkIndex: Int, tolerance: Double, debug: Boolean): Boolean {
    input.forEachIndexed { index, row ->
        val result = this.test(create(row))
        val expected = create(output[index])
        if (abs(result - expected).any { it > tolerance }) {
            if (debug) println("network: $networkIndex\t\texpected: $expected\t\tgot: $result")
            return false
        }
    }
    return true
}

fun StandardNN.computeAverageError(input: List<DoubleArray>, output: List<DoubleArray>): Double =
    input.mapIndexed { index, row ->
        val result = this.test(create(row))
        val expected = create(output[index])
        abs(result - expected).elementSum() / result.numCols()
    }.sum() / input.size


enum class Dataset(
        private val scaleOptions: ScaleOptions,
        private val filename: String,
        private val tolerance: Double
) {
    XOR(ScaleOptions.DO_NOT_SCALE, "test_data1.txt", 0.05),
    CHEMICAL(ScaleOptions.SCALE_SEPARATELY, "chemical.txt", 0.05),
    ADD(ScaleOptions.SCALE_TOGETHER, "add.txt", 0.05);

    fun doTest(
            testIndex: Int,
            actFun: SimpleActivationFunction,
            addSamples: Int = 5,
            testSampleSize: Int = 5,
            debug: Boolean = false
    ): Array<Triple<Boolean, Int, Double>> {

        println("Executing test number ${testIndex + 1}...")

        val lines = File(filename).readLines()
        val topology = lines.first().split(" ").map { it.toInt() }
        val hiddenCounts = topology.subList(1, topology.lastIndex)

        val inputList = lines.asSequence().filter { it.startsWith("in: ") }.map { input -> input.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList()
        val outputList = lines.asSequence().filter { it.startsWith("out: ") }.map { output -> output.split(" ").asSequence().drop(1).map { it.toDouble() }.toList().toDoubleArray() }.toList()

        when (scaleOptions) {
            ScaleOptions.DO_NOT_SCALE -> {
                // do nothing
            }
            ScaleOptions.SCALE_SEPARATELY -> {
                val inputBounds = create(inputList.toTypedArray()).T.to2DArray().map { Pair(it.min()!!, it.max()!!) }
                val outputBounds = create(outputList.toTypedArray()).T.to2DArray().map { Pair(it.min()!!, it.max()!!) }

                inputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - inputBounds[it].first) / (inputBounds[it].second - inputBounds[it].first) } }
                outputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - outputBounds[it].first) / (outputBounds[it].second - outputBounds[it].first) } }
            }
            ScaleOptions.SCALE_TOGETHER -> {
                val min = Math.min(inputList.map { it.min()!! }.min()!!, outputList.map { it.min()!! }.min()!!)
                val max = Math.max(inputList.map { it.max()!! }.max()!!, outputList.map { it.max()!! }.max()!!)

                inputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - min) / (max - min) } }
                outputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - min) / (max - min) } }
            }
        }

        val trainInputList = inputList.subList(0, inputList.size - testSampleSize)
        val trainOutputList = outputList.subList(0, outputList.size - testSampleSize)

        val testInputList = inputList.subList(inputList.size - testSampleSize, inputList.size)
        val testOutputList = outputList.subList(outputList.size - testSampleSize, outputList.size)

        val inputMatrix = create(trainInputList.toTypedArray())
        val outputMatrix = create(trainOutputList.toTypedArray())

        val inputBuckets = trainInputList.chunked(addSamples).map { create(it.toTypedArray()) }
        val outputsBuckets = trainOutputList.chunked(addSamples).map { create(it.toTypedArray()) }

        val networks = arrayOf(
                SplineNN(hiddenCounts = hiddenCounts, splineInitFunction = actFun),
                C2SplineNN2(hiddenCounts = hiddenCounts, splineInitFunction = actFun),
                StandardNN(hiddenCounts = hiddenCounts, activationFunction = actFun)
//                C2SplineNNBaseMatrix(hiddenCounts = hiddenCounts, splineInitFunction = actFun)
        )

        val trained = Array(networks.size) {Triple(false, false, 0)}

        networks[0].initialize(inputMatrix, outputMatrix, InitializationMethod.GLOROT)

        for (i in (1 until networks.size)) {
            networks[i].weights = networks[0].weights.map { it.copy() }
            networks[i].initialize(inputMatrix, outputMatrix, InitializationMethod.CUSTOM)
        }

        for (i in inputBuckets.indices) {
            networks.forEachIndexed { index, network ->
                if (!trained[index].second) {
                    if (debug && trained[index].first) println("$index : first pass")
                    network.train(inputBuckets[i], outputsBuckets[i])
                    val passed = network.test(testInputList, testOutputList, index, tolerance, debug)
                    trained[index] = Triple(passed, trained[index].first && passed,trained[index].third + inputBuckets[i].numRows())
                }
            }
        }

        val results = trained.mapIndexed { index, triple -> Triple(triple.second, triple.third, networks[index].computeAverageError(testInputList, testOutputList)) }.toTypedArray()
        if (debug) results.forEachIndexed { index, triple ->  println("$index - $triple") }
        return results
    }

    enum class ScaleOptions {
        DO_NOT_SCALE, SCALE_SEPARATELY, SCALE_TOGETHER
    }
}