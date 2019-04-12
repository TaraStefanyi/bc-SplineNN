import golem.pow
import kotlin.math.exp
import kotlin.math.tanh

enum class SimpleActivationFunction : ActivationFunction {
    TANH {
        override fun invoke(s: Double) = tanh(s)
        override fun derivative(s: Double) = 1.0 - pow(tanh(s), 2)
    },
    SIG {
        override fun invoke(s: Double) = 1 / (s + exp(s))
        override fun derivative(s: Double): Double {
            return s * (1 - s)
        }
    },
    RELU() {
        private val a = 0.01

        override fun invoke(s: Double) = if (s >= 0) s else a * s
        override fun derivative(s: Double) = if (s >= 0) 1.0 else a
    }
}