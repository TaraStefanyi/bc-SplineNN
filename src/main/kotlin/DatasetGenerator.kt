import java.io.File
import java.util.*
import kotlin.random.Random

fun main() {
    generateAdditionDataset(arrayOf(2, 2, 1).toIntArray(), 500, "add.txt")
}

fun generateAdditionDataset(topology: IntArray, samples: Int, filename: String) {
    val file = File(filename)
    assert(!file.exists())

    file.createNewFile()
    file.writeText(topology.joinToString(" "))
    file.appendText("\n")

    val random = Random(Date().time)

    var a: Int
    var b: Int

    (0 until samples).forEach {
        a = random.nextInt(1000)
        b = random.nextInt(1000)
        file.appendText("in: $a $b\nout: ${a+b}\n")
    }
}