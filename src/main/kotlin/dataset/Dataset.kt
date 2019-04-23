package dataset

import ActivationFunction
import network.*

interface Dataset {
    var data: Data?

    fun loadDataset(): Data

    fun doTest(
            testIndex: Int,
            actFun: ActivationFunction,
            debug: Boolean = false
    ): Array<TestResult> {

        println("Executing test number ${testIndex + 1}...")

        val data = loadDataset()

        val networks = arrayOf(
                SplineNN(hiddenCounts = data.hiddenCounts, splineInitFunction = actFun),
                C2SplineNN(hiddenCounts = data.hiddenCounts, splineInitFunction = actFun),
                StandardNN(hiddenCounts = data.hiddenCounts, activationFunction = actFun)
        )

        val trained = Array(networks.size) { Triple(false, false, 0) }

        networks[0].initialize(data.inputMatrix, data.outputMatrix, InitializationMethod.GLOROT)

        for (i in (1 until networks.size)) {
            networks[i].weights = networks[0].weights.map { it.copy() }
            networks[i].initialize(data.inputMatrix, data.outputMatrix, InitializationMethod.CUSTOM)
        }

        for (i in data.inputBuckets.indices) {
            networks.forEachIndexed { index, network ->
                if (!trained[index].second) {
                    if (debug && trained[index].first) println("$index : first pass")
                    network.train(data.inputBuckets[i], data.outputsBuckets[i])
                    val passed = network.test(data.testInputList, data.testOutputList, index, data.tolerance, debug, data.considerNegativeAsZero)
                    trained[index] = Triple(passed, trained[index].first && passed, trained[index].third + data.inputBuckets[i].numRows())
                }
            }
        }

        val results = trained.mapIndexed { index, triple -> TestResult(networks[index]::class.java.simpleName, triple.second, triple.third, networks[index].computeAverageError(data.testInputList, data.testOutputList)) }.toTypedArray()
        if (debug) results.forEachIndexed { index, triple -> println("$index - $triple") }
        return results
    }
}