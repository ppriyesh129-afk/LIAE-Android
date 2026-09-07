package com.liae.android

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class LiaeUdEngine(context: Context) {

    companion object {
        private const val MODEL_NAME =
            "LIAE_128_80_48_16_fp32.onnx"

        private const val SIZE = 128
        private const val CHANNELS = 3
    }

    private val environment =
        OrtEnvironment.getEnvironment()

    private val session: OrtSession

    init {
        val modelFile =
            context.getFileStreamPath(MODEL_NAME)

        if (!modelFile.exists() ||
            modelFile.length() < 10_000_000L
        ) {
            context.assets.open(MODEL_NAME).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(
                        output,
                        DEFAULT_BUFFER_SIZE
                    )
                }
            }
        }

        if (!modelFile.exists()) {
            throw IllegalStateException(
                "LIAE model file was not created"
            )
        }

        if (modelFile.length() < 10_000_000L) {
            throw IllegalStateException(
                "LIAE model is too small: " +
                    modelFile.length() +
                    " bytes"
            )
        }

        val options =
            OrtSession.SessionOptions()

        session =
            environment.createSession(
                modelFile.absolutePath,
                options
            )
    }

    data class Result(
        val rgb: FloatArray,
        val mask: FloatArray,
        val rgbMin: Float,
        val rgbMax: Float,
        val rgbMean: Float,
        val maskMin: Float,
        val maskMax: Float,
        val maskMean: Float
    )

    fun run(
        src: FloatArray,
        dst: FloatArray
    ): Result {

        require(
            src.size ==
                SIZE * SIZE * CHANNELS
        ) {
            "src must contain 128x128x3 floats"
        }

        require(
            dst.size ==
                SIZE * SIZE * CHANNELS
        ) {
            "dst must contain 128x128x3 floats"
        }

        val srcTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(src),
                longArrayOf(
                    1,
                    SIZE.toLong(),
                    SIZE.toLong(),
                    CHANNELS.toLong()
                )
            )

        val dstTensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(dst),
                longArrayOf(
                    1,
                    SIZE.toLong(),
                    SIZE.toLong(),
                    CHANNELS.toLong()
                )
            )

        try {

            val inputs =
                mapOf(
                    "src" to srcTensor,
                    "dst" to dstTensor
                )

            session.run(inputs).use { output ->

                val rgb =
                    flattenTensor(
                        output[0].value
                    )

                val mask =
                    flattenTensor(
                        output[1].value
                    )

                val rgbStats =
                    statistics(rgb)

                val maskStats =
                    statistics(mask)

                return Result(
                    rgb = rgb,
                    mask = mask,

                    rgbMin = rgbStats.min,
                    rgbMax = rgbStats.max,
                    rgbMean = rgbStats.mean,

                    maskMin = maskStats.min,
                    maskMax = maskStats.max,
                    maskMean = maskStats.mean
                )
            }

        } finally {
            srcTensor.close()
            dstTensor.close()
        }
    }

    private data class Stats(
        val min: Float,
        val max: Float,
        val mean: Float
    )

    private fun statistics(
        values: FloatArray
    ): Stats {

        require(values.isNotEmpty())

        var min =
            Float.POSITIVE_INFINITY

        var max =
            Float.NEGATIVE_INFINITY

        var sum = 0.0

        for (value in values) {

            if (value < min) {
                min = value
            }

            if (value > max) {
                max = value
            }

            sum += value.toDouble()
        }

        return Stats(
            min = min,
            max = max,
            mean =
                (sum / values.size)
                    .toFloat()
        )
    }

    private fun flattenTensor(
        value: Any
    ): FloatArray {

        return when (value) {

            is FloatArray -> {
                value.copyOf()
            }

            is Array<*> -> {

                val values =
                    ArrayList<Float>()

                fun collect(item: Any?) {

                    when (item) {

                        is FloatArray -> {

                            for (v in item) {
                                values.add(v)
                            }
                        }

                        is Array<*> -> {

                            for (child in item) {
                                collect(child)
                            }
                        }
                    }
                }

                collect(value)

                FloatArray(
                    values.size
                ) { index ->
                    values[index]
                }
            }

            else -> {

                throw IllegalArgumentException(
                    "Unsupported ONNX output type: " +
                        value::class.java.name
                )
            }
        }
    }

    fun close() {
        session.close()
    }
}

Then replace the result section in "MainActivity.kt".

Find:

statusText.text =
    """
    LIAE inference complete

    RGB: ${result.rgb.size}
    Mask: ${result.mask.size}
    """.trimIndent()

Replace it with:

statusText.text =
    """
    LIAE inference complete

    RGB: ${result.rgb.size}
    Mask: ${result.mask.size}

    RGB min: ${result.rgbMin}
    RGB max: ${result.rgbMax}
    RGB mean: ${result.rgbMean}

    Mask min: ${result.maskMin}
    Mask max: ${result.maskMax}
    Mask mean: ${result.maskMean}
    """.trimIndent()

Then

Commit both files and build the APK again.

After installing it, select the same image and send me exactly what it shows for:

RGB min:
RGB max:
RGB mean:

Mask min:
Mask max:
Mask mean:

Don't change "ImageTensor.kt" yet.

Those six numbers will tell us exactly how the LIAE ONNX output needs to be converted to an Android bitmap.
