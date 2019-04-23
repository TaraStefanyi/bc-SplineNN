import dataset.*

fun main() {
    val runs = 1
    val debug = true
    val actFun = SimpleActivationFunction.SIG
    val dataset = MnistDataset()

    val testResults = List(runs) {dataset.doTest(it, actFun, debug = debug)}

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
}
