package dataset

import golem.create
import java.io.IOException
import java.io.InputStream
import java.io.FileInputStream


class MnistDataset(
        private val tolerance: Double = 0.05,
        private val addSamples: Int = 1000
) : Dataset {
    override var data: Data? = null

    override fun loadDataset(): Data {
        val (trainInputList, trainOutputList) = readDataFiles("datasets/mnist/train-images", "datasets/mnist/train-labels")
        val (testInputList, testOutputList) = readDataFiles("datasets/mnist/test-images", "datasets/mnist/test-labels")

        this.data = Data(
                hiddenCounts = listOf(30),
                tolerance = tolerance,
                considerNegativeAsZero = true,
                inputMatrix = create(trainInputList.toTypedArray()),
                outputMatrix = create(trainOutputList.toTypedArray()),
                inputBuckets = trainInputList.chunked(addSamples).map { create(it.toTypedArray()) },
                outputsBuckets = trainOutputList.chunked(addSamples).map { create(it.toTypedArray()) },
                testInputList = testInputList,
                testOutputList = testOutputList
        )
        return this.data!!
    }

    @Throws(IOException::class)
    private fun readInt(inputStream: InputStream): Int {
        val b = IntArray(4) {inputStream.read()}
        return b[3] or (b[2] shl 8) or (b[1] shl 16) or (b[0] shl 24)
    }

    @Throws(IOException::class)
    private fun readDataFiles(imageFile: String, labelFile: String) = Pair(
        FileInputStream(imageFile).use { inputStream ->
            val magic = readInt(inputStream)
            val totalImages = readInt(inputStream)
            val totalRows = readInt(inputStream)
            val totalCols = readInt(inputStream)
            List(totalImages) { DoubleArray(totalRows * totalCols) { inputStream.read().toDouble() } }
        },

        FileInputStream(labelFile).use { inputStream ->
            val magic = readInt(inputStream)
            val totalLabels = readInt(inputStream)
            val labelData = IntArray(totalLabels) {inputStream.read()}
            List(totalLabels) { i -> DoubleArray(10) {if (labelData[i] == it) 1.0 else 0.0} }
        }
    )
}