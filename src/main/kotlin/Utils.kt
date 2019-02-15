import golem.*
import golem.matrix.Matrix

val LAMBDA = pow(10.0, -5)
//val LAMBDA = pow(10.0, -4)
val LAMBDA_Q0 = pow(10.0, -5)
//val LAMBDA_Q0 = pow(10.0, -1)
const val INT = 0.1
const val EXT = 3.0
const val MAX = 20
const val RATIO = 10
const val SIG = 0.1
const val RHO = SIG / 2

fun Matrix<Double>.addBiasColumn(): Matrix<Double> {
    val result = ones(this.numRows(), this.numCols()+1)
    result[0 until this.numRows(), 0 until this.numCols()] = this
    result.setCol(this.numCols(), fill(this.numRows(), 1, 1.0))
    return result
}

fun flattenMatrixListVertically(list: List<Matrix<Double>>): Matrix<Double> {
    return create(list.flatMap { it.T.toList() }.toDoubleArray()).T
}

fun Matrix<Double>.dropLastRow(): Matrix<Double> {
    return create(this.to2DArray().dropLast(1).toTypedArray())
}

fun Matrix<Double>.toSingleColumn(): Matrix<Double> {
    return create(this.T.toList().toDoubleArray()).T
}

fun copyColumnVectorHorizontally(column: DoubleArray, times: Int): Matrix<Double> {
    return create(Array(times) {column}).T
}

fun <T> Matrix<T>.sub2ind(rowSub: Int, colSub: Int): Int {
    return colSub*numRows() + rowSub
}

fun MutableList<Int>.customAdd(i: Int) {
    add(i)
}

fun <T> Matrix<T>.mapRowsIndexed(f: (i: Int, Matrix<T>) -> Matrix<T>): Matrix<T> {
    val outRows = Array(this.numRows()) {
        f(it, this.getRow(it))
    }

    val out = this.getFactory().zeros(this.numRows(), outRows[0].numCols())

    outRows.forEachIndexed { i, matrix ->
        if (matrix.numCols() != out.numCols())
            throw RuntimeException("All output rows of mapRows must have same number of columns")
        else
            out.setRow(i, outRows[i])
    }
    return out
}

fun <T> Matrix<T>.mapColsIndexed(f: (i: Int, Matrix<T>) -> Matrix<T>): Matrix<T> {

    val outCols = Array(this.numCols()) {
        val out = f(it, this.getCol(it))
        // If user creates a row vector auto convert to column for them
        if (out.numRows() == 1) out.T else out
    }

    val out = this.getFactory().zeros(outCols[0].numRows(), this.numCols())

    outCols.forEachIndexed { i, matrix ->
        if (matrix.numRows() != out.numRows())
            throw RuntimeException("All output rows of mapCols must have same number of columns")
        else
            out.setCol(i, outCols[i])
    }
    return out
}

fun <T> Matrix<T>.eachRowIndexed(f: (i: Int, Matrix<T>) -> Unit) {
    for (row in 0 until this.numRows())
        f(row, this.getRow(row))
}

fun <T, U> Matrix<T>.mapRowsToListIndexed(f: (i: Int, Matrix<T>) -> U): List<U> {
    val a = ArrayList<U>(this.numRows())
    this.eachRowIndexed { i, row ->
        a.add(f(i, row))
    }
    return a
}