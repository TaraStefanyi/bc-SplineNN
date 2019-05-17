import dataset.*
import golem.plot

fun main() {
    val runs = 1000
    val debug = false
    val actFun = SimpleActivationFunction.RELU
    val dataset = XorDataset()

    val testResults = List(runs) {dataset.doTest(it, actFun, 10, 1000, debug = debug)}
//    val testResults = List(runs) {dataset.doTestBuckets(it, actFun, 300, debug = debug)}

    val networks = testResults.first().map { it.network }

    val successfulRuns = testResults
            .map { results -> results.map { if (it.success) 1 else 0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .map { it }

    val successRate = successfulRuns.map { it.toDouble() / runs }

    val averageErrorSuccess = testResults
            .map { results -> results.map { if (it.success) it.averageError else 0.0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .mapIndexed { i, totalError -> totalError / successfulRuns[i] }

    val averageErrorFail = testResults
            .map { results -> results.map { if (!it.success) it.averageError else 0.0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .mapIndexed { i, totalError -> totalError / (runs - successfulRuns[i]) }

    val averageSamplesUsedSuccess = testResults
            .map { results -> results.map { if (it.success) it.samplesUsed else 0 } }
            .reduce { acc, list -> acc.zip(list) {a, b -> a + b} }
            .mapIndexed { i, totalSamplesUsed -> totalSamplesUsed.toDouble() / successfulRuns[i] }


    println("Networks tested: $networks")
    println("Successful runs: $successfulRuns")
    println("Success rate: $successRate")
    println("Average error on success: $averageErrorSuccess")
    println("Average error on fail: $averageErrorFail")
    println("Average samples used on success: $averageSamplesUsedSuccess")

    val colors = arrayOf("r", "b", "g")

//    testResults[0].forEachIndexed { i, result -> plot(null, result.lossFunction.toDoubleArray(), color = colors[i], lineLabel = result.network) }

    val averageLossFunction = testResults
            .map { results -> results.map { it.lossFunction } }
            .reduce { acc, list -> acc.zip(list) {a, b ->
                (0 until maxOf(a.size, b.size)).map { a.getOrElse(it) {0.0} + b.getOrElse(it) {0.0} }
            } }
            .map { sum -> sum.map { it / runs } }

    averageLossFunction.forEachIndexed { i, result -> plot(null, result.toDoubleArray(), color = colors[i], lineLabel = networks[i]) }
}
