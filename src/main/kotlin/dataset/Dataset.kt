package dataset

import ActivationFunction
import network.*

abstract class Dataset {
    val data by lazy { loadDataset() }

    protected abstract fun loadDataset(): Data

    fun doTest(
            testIndex: Int,
            actFun: ActivationFunction,
            addEpochs: Int,
            maxIterations: Int,
            debug: Boolean = false
    ): Array<TestResult> {

        println("Executing test number ${testIndex + 1}...")

        val networks = initNetworks(actFun)

        val trained = Array(networks.size) { Triple(false, false, 0) }

        val loss = Array(networks.size) { mutableListOf<Double>()}

        for (i in 0 until maxIterations) {
            networks.forEachIndexed { index, network ->
                if (!trained[index].second) {
                    if (debug && trained[index].first) println("$index : first pass")
                    loss[index].addAll(network.train(data.inputMatrix, data.outputMatrix, addEpochs))
                    val passed = network.test(data.testInputList, data.testOutputList, index, data.tolerance, debug, data.considerNegativeAsZero)
                    trained[index] = Triple(passed, trained[index].first && passed, trained[index].third + addEpochs)
                }
            }
        }

        val results = trained.mapIndexed { index, triple -> TestResult(networks[index]::class.java.simpleName, triple.second, triple.third, networks[index].computeAverageError(data.testInputList, data.testOutputList), loss[index]) }.toTypedArray()
        if (debug) results.forEachIndexed { index, triple -> println("$index - $triple") }
        return results
    }

    fun doTestBuckets(
            testIndex: Int,
            actFun: ActivationFunction,
            epochsWithEachBucket: Int,
            debug: Boolean = false
    ): Array<TestResult> {

        println("Executing test number ${testIndex + 1}...")

        val networks = initNetworks(actFun)

        val trained = Array(networks.size) { Triple(false, false, 0) }

        val loss = Array(networks.size) { mutableListOf<Double>()}

        for (i in data.inputBuckets.indices) {
            networks.forEachIndexed { index, network ->
                if (!trained[index].second) {
                    if (debug && trained[index].first) println("$index : first pass")
                    loss[index].addAll(network.train(data.inputBuckets[i], data.outputsBuckets[i], epochsWithEachBucket))
                    val passed = network.test(data.testInputList, data.testOutputList, index, data.tolerance, debug, data.considerNegativeAsZero)
                    trained[index] = Triple(passed, trained[index].first && passed, trained[index].third + data.inputBuckets[i].numRows())
                }
            }
        }

        val results = trained.mapIndexed { index, triple -> TestResult(networks[index]::class.java.simpleName, triple.second, triple.third, networks[index].computeAverageError(data.testInputList, data.testOutputList), loss[index]) }.toTypedArray()
        if (debug) results.forEachIndexed { index, triple -> println("$index - $triple") }
        return results
    }

    private fun initNetworks(actFun: ActivationFunction): Array<StandardNN> {
        val networks = arrayOf(
                SplineNN(hiddenCounts = data.hiddenCounts, splineInitFunction = actFun),
                C2SplineNN(hiddenCounts = data.hiddenCounts, splineInitFunction = actFun),
                StandardNN(hiddenCounts = data.hiddenCounts, activationFunction = actFun)
        )

        networks[0].initialize(data.inputMatrix, data.outputMatrix, InitializationMethod.GLOROT)

        for (i in (1 until networks.size)) {
            networks[i].weights = networks[0].weights.map { it.copy() }
            networks[i].initialize(data.inputMatrix, data.outputMatrix, InitializationMethod.CUSTOM)
        }

        return networks
    }

}