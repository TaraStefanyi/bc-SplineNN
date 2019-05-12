package dataset

import golem.create
import java.io.File

abstract class CustomDataset(
        private val scaler: Scaler,
        private val filename: String,
        private val tolerance: Double,
        private val addSamples: Int = 5,
        private val testSampleSize: Int = 5
) : Dataset() {
    override fun loadDataset(): Data {
        val lines = File(filename).readLines()
        val topology = lines.first().split(" ").map { it.toInt() }

        val inputList = lines
                .asSequence()
                .filter { it.startsWith("in: ") }
                .map { input ->
                    input
                            .split(" ")
                            .asSequence()
                            .drop(1)
                            .map { it.toDouble() }
                            .toList()
                            .toDoubleArray()
                }
                .toList()

        val outputList = lines
                .asSequence()
                .filter { it.startsWith("out: ") }
                .map { output ->
                    output
                            .split(" ")
                            .asSequence()
                            .drop(1)
                            .map { it.toDouble() }
                            .toList()
                            .toDoubleArray()
                }
                .toList()

        scaler.scale(inputList, outputList)

        val trainInputList = inputList.subList(0, inputList.size - testSampleSize)
        val trainOutputList = outputList.subList(0, outputList.size - testSampleSize)

        return Data(
                hiddenCounts = topology.subList(1, topology.lastIndex),
                tolerance = tolerance,
                considerNegativeAsZero = false,
                inputMatrix = create(trainInputList.toTypedArray()),
                outputMatrix = create(trainOutputList.toTypedArray()),
                inputBuckets = trainInputList.chunked(addSamples).map { create(it.toTypedArray()) },
                outputsBuckets = trainOutputList.chunked(addSamples).map { create(it.toTypedArray()) },
                testInputList = inputList.subList(inputList.size - testSampleSize, inputList.size),
                testOutputList = outputList.subList(outputList.size - testSampleSize, outputList.size)
        )
    }
}