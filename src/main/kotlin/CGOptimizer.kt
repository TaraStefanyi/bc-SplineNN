import golem.*
import golem.matrix.Matrix

class CGOptimizer: Optimizer {
    override fun optimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>) {
        val results = minimize(network, weights, inputs, outputs, List(1) {500})
        val lastNonZeroIdx = results.fval.indexOfLast { it != 0.0 }
        val lastIndex = results.fval.toList().lastIndex
        if (lastNonZeroIdx < lastIndex) {
            results.fval[lastNonZeroIdx + 1..lastIndex, 0] = results.fval[lastNonZeroIdx, 0]
            results.gval[lastNonZeroIdx + 1..lastIndex, 0] = results.gval[lastNonZeroIdx, 0]
        }
        network.reshapeWeightFromVector(results.weights)
    }

    private fun minimize(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, outputs: Matrix<Double>, length: List<Int>): MinimizeResults {
        var x1: Double; var x2: Double; var x3: Double; var x4 = 0.0
        var d0: Double; var d1: Double; var d2: Double; var d3: Double; var d4 = 0.0
        var f1: Double; var f2: Double; var f3: Double; var f4 = 0.0
        var a: Double; var b: Double
        var x0Copy: Matrix<Double>; var f0Copy: Double; var df0Copy: Matrix<Double>
        var m: Int
        var df3: Matrix<Double>
        var x = weights.copy()

        var red = 1
        if (length.size == 2) red = length[1] else if (length.size != 1) throw IllegalArgumentException()
        val len = length[0]
        var i = 0
        var lsFailed = true
        var (f0, df0) = backPropagation(network, x, inputs, outputs)

        val fX = zeros(len, 1)
        val dfX = zeros(len, 1)

        fX[0] = f0
        dfX[0] = df0.norm()
        i++
        var s = -df0
        d0 = (-s.T * s)[0]
        x3 = red/(1.0 - d0)

        while (i < len) {
            x0Copy = x.copy()
            f0Copy = f0
            df0Copy = df0.copy()
            m = min(MAX, len-i)

            while (true) {
                x2 = 0.0
                f2 = f0
                f3 = f0
                d2 = d0
                df3 = df0

                while (m > 0) {
                    m--
                    i++
                    val bpResults = backPropagation(network, x + (s * x3), inputs, outputs)
                    f3 = bpResults.fval
                    df3 = bpResults.grads
                    if (f3.isNaN() || f3.isInfinite() || df3.any { it.isInfinite() || it.isNaN() }) {
                        x3 = (x2+x3)/2
                    } else {
                        break
                    }
                }

                if (f3 < f0Copy) {
                    x0Copy = x + s * x3
                    f0Copy = f3
                    df0Copy = df3
                }

                d3 = (df3.T * s)[0]

                if ((d3 > d0*SIG) || (f3 > f0+x3*RHO*d0) || (m == 0)){
                    break
                }

                x1 = x2
                f1 = f2
                d1 = d2

                x2 = x3
                f2 = f3
                d2 = d3

                a = 6*(f1-f2)+3*(d2+d1)*(x2-x1)
                b = 3*(f2-f1)-(2*d1+d2)*(x2-x1)

                x3 = x1 - d1 * pow(x2 - x1, 2) / (b + sqrt(b * b - a * d1 * (x2 - x1)))

                if (x3.isNaN() || x3.isInfinite() || x3 < 0) {
                    x3 = x2*EXT
                } else if (x3 > x2*EXT) {
                    x3 = x2*EXT
                } else if (x3 < x2+INT*(x2-x1)) {
                    x3 = x2+INT*(x2-x1)
                }
            }


            //interpolation
            while ((abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && m > 0) {
                if (d3 > 0 || f3 > f0+x3*RHO*d0) {
                    x4 = x3
                    f4 = f3
                    d4 = d3
                } else {
                    x2 = x3
                    f2 = f3
                    d2 = d3
                }

                if (f4 > f0) {
                    x3 = x2-(0.5*d2*pow(x4-x2, 2))/(f4-f2-d2*(x4-x2))
                } else {
                    a = 6*(f2-f4)/(x4-x2)+3*(d4+d2)
                    b = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                    x3 = x2+(sqrt(b*b-a*d2*pow(x4-x2, 2))-b)/a
                    if (x3.isNaN() || x3.isInfinite()) {
                        x3 = (x2+x4)/2
                    }
                }

                x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
                val bpResults = backPropagation(network, x + (s * x3), inputs, outputs)
                f3 = bpResults.fval
                df3 = bpResults.grads

                if (f3 < f0Copy) {
                    x0Copy = x+x3*s
                    f0Copy = f3
                    df0Copy = df3
                }

                m--
                i++
                d3 = (df3.T * s)[0]
            }


            if ((abs(d3) < -SIG*d0) && (f3 < f0+x3*RHO*d0)) {
                x += (x3 * s)
                f0 = f3
                val lastNonZeroIndex = fX.indexOfLast { it != 0.0 }
                fX[lastNonZeroIndex+1 until i, 0] = f3
                dfX[lastNonZeroIndex+1 until i, 0] = df3.norm()
                s = ((df3.T*df3)[0] - (df0.T*df3)[0])/(df0.T*df0)[0]*s - df3
                df0 = df3
                d3 = d0
                d0 = (df0.T*s)[0]
                if (d0 > 0) {
                    s = -df0
                    d0 = (-s.T*s)[0]
                }
                x3 *= min(RATIO, d3 / (d0 - Double.MIN_VALUE))
                lsFailed = false
            } else {
                x = x0Copy
                f0 = f0Copy
                df0 = df0Copy
                if (lsFailed || i > abs(len)) break
                s = -df0
                d0 = (-s.T*s)[0]
                x3 = 1/(1-d0)
                lsFailed = true
            }
        }

        return MinimizeResults(x, fX, dfX)
    }

    private fun backPropagation(network: StandardNN, weights: Matrix<Double>, inputs: Matrix<Double>, expectedOutputs: Matrix<Double>): BackPropagationResults {
        network.reshapeWeightFromVector(weights)
        val computedOutputs = network.passForward(inputs)
        val fval = network.computeObjectiveFunction(expectedOutputs, computedOutputs)
        val gradients = network.passBackward(inputs, expectedOutputs, computedOutputs)
        val g = gradients.gradsError + gradients.gradsWeights
        return BackPropagationResults(fval, g)
    }

    private data class BackPropagationResults(val fval: Double, val grads: Matrix<Double>)
    private data class MinimizeResults(val weights: Matrix<Double>, val fval: Matrix<Double>, val gval: Matrix<Double>)
}