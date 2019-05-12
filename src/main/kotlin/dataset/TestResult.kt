package dataset

data class TestResult(
        val network: String,
        val success: Boolean,
        val samplesUsed: Int,
        val averageError: Double,
        val lossFunction: List<Double>
)