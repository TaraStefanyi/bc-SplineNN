interface ActivationFunction {
    operator fun invoke(s :Double) :Double
    fun derivative (s :Double) :Double
}