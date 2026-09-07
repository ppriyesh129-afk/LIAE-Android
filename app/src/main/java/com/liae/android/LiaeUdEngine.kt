package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

class LiaeUdEngine(context: Context) {

    companion object {
        private const val MODEL_NAME = "LIAE_128_80_48_16_fp32.onnx"

        private const val SIZE = 128
        private const val CHANNELS = 3

        private const val IMAGE_FLOATS = SIZE * SIZE * CHANNELS
        private const val MASK_FLOATS = SIZE * SIZE
    }

    data class Result(
        val rgb: FloatArray,
        val dstMask: FloatArray,
        val srcMask: FloatArray,
        val debugText: String
    )

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {

        val modelFile = context.getFileStreamPath(MODEL_NAME)

        if (!modelFile.exists()) {
            context.assets.open(MODEL_NAME).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        session = env.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions()
        )

        require(session.inputNames == setOf("in_face")) {
            "Expected ONNX input 'in_face', got ${session.inputNames}"
        }

        require(
            session.outputNames.containsAll(
                setOf("output_1", "output_2", "output_3")
            )
        ) {
            "Unexpected ONNX outputs: ${session.outputNames}"
        }
    }

    fun run(dst: FloatArray): Result {

        require(dst.size == IMAGE_FLOATS)

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(dst),
            longArrayOf(1, SIZE.toLong(), SIZE.toLong(), CHANNELS.toLong())
        )

        return try {

            session.run(
                mapOf("in_face" to inputTensor)
            ).use { outputs ->

                val dstMaskTensor =
                    outputs["output_1"]?.get() as? OnnxTensor
                        ?: error("Missing output_1")

                val rgbTensor =
                    outputs["output_2"]?.get() as? OnnxTensor
                        ?: error("Missing output_2")

                val srcMaskTensor =
                    outputs["output_3"]?.get() as? OnnxTensor
                        ?: error("Missing output_3")

                val rgb = extractRgb(rgbTensor.value)
                val dstMask = extractMask(dstMaskTensor.value)
                val srcMask = extractMask(srcMaskTensor.value)

                val rgbInfo = rgbTensor.info
                val maskInfo = dstMaskTensor.info
                val srcInfo = srcMaskTensor.info

                val debugText = buildString {

                    appendLine("LIAE DEBUG")
                    appendLine()

                    appendLine("INPUTS")
                    appendLine(session.inputNames.joinToString())
                    appendLine()

                    appendLine("OUTPUTS")
                    appendLine(session.outputNames.joinToString())
                    appendLine()

                    appendLine("RGB INFO")
                    appendLine("shape=${rgbInfo.shape.contentToString()}")
                    appendLine("type=${rgbInfo.type}")
                    appendLine()

                    appendLine("DST MASK INFO")
                    appendLine("shape=${maskInfo.shape.contentToString()}")
                    appendLine("type=${maskInfo.type}")
                    appendLine()

                    appendLine("SRC MASK INFO")
                    appendLine("shape=${srcInfo.shape.contentToString()}")
                    appendLine("type=${srcInfo.type}")
                    appendLine()

                    appendLine("RGB")
                    appendLine("min=${rgb.minOrNull()}")
                    appendLine("max=${rgb.maxOrNull()}")
                    appendLine("mean=${rgb.average()}")
                    appendLine()

                    appendLine("DST MASK")
                    appendLine("min=${dstMask.minOrNull()}")
                    appendLine("max=${dstMask.maxOrNull()}")
                    appendLine("mean=${dstMask.average()}")
                    appendLine()

                    appendLine("SRC MASK")
                    appendLine("min=${srcMask.minOrNull()}")
                    appendLine("max=${srcMask.maxOrNull()}")
                    appendLine("mean=${srcMask.average()}")
                }

                Result(
                    rgb = rgb,
                    dstMask = dstMask,
                    srcMask = srcMask,
                    debugText = debugText
                )
            }

        } finally {
            inputTensor.close()
        }
    }

    private fun extractRgb(value: Any): FloatArray {

        val tensor =
            value as Array<Array<Array<FloatArray>>>

        val batch = tensor[0]

        val out = FloatArray(IMAGE_FLOATS)

        var i = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val p = batch[y][x]

                out[i++] = p[0]
                out[i++] = p[1]
                out[i++] = p[2]
            }
        }

        return out
    }

    private fun extractMask(value: Any): FloatArray {

        val tensor =
            value as Array<Array<Array<FloatArray>>>

        val batch = tensor[0]

        val out = FloatArray(MASK_FLOATS)

        var i = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                out[i++] = batch[y][x][0]
            }
        }

        return out
    }

    fun close() {
        session.close()
    }
}
