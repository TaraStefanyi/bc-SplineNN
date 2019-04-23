package dataset

import golem.create
import golem.to2DArray

enum class Scaler {
    DO_NOT_SCALE {
        override fun scale(inputList: List<DoubleArray>, outputList: List<DoubleArray>) {
            // do nothing
        }
    }, SCALE_SEPARATELY {
        override fun scale(inputList: List<DoubleArray>, outputList: List<DoubleArray>) {
            val inputBounds = create(inputList.toTypedArray()).T.to2DArray().map { Pair(it.min()!!, it.max()!!) }
            val outputBounds = create(outputList.toTypedArray()).T.to2DArray().map { Pair(it.min()!!, it.max()!!) }

            inputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - inputBounds[it].first) / (inputBounds[it].second - inputBounds[it].first) } }
            outputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - outputBounds[it].first) / (outputBounds[it].second - outputBounds[it].first) } }
        }
    }, SCALE_TOGETHER {
        override fun scale(inputList: List<DoubleArray>, outputList: List<DoubleArray>) {
            val min = Math.min(inputList.map { it.min()!! }.min()!!, outputList.map { it.min()!! }.min()!!)
            val max = Math.max(inputList.map { it.max()!! }.max()!!, outputList.map { it.max()!! }.max()!!)

            inputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - min) / (max - min) } }
            outputList.forEach { row -> (0 until row.size).forEach { row[it] = (row[it] - min) / (max - min) } }
        }
    };

    abstract fun scale(inputList: List<DoubleArray>, outputList: List<DoubleArray>)
}