package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.util.Locale

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

        require(dst.size == IMAGE_FLOATS) {
            "Expected $IMAGE_FLOATS floats, got ${dst.size}"
        }

        val inputShape = longArrayOf(
            1,
            SIZE.toLong(),
            SIZE.toLong(),
            CHANNELS.toLong()
        )

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(dst),
            inputShape
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

                val rgb =
                    extractRgb(rgbTensor.value)

                val dstMask =
                    extractMask(dstMaskTensor.value)

                val srcMask =
                    extractMask(srcMaskTensor.value)

                val debugText =
                    String.format(
                        Locale.US,
                        "LIAE DEBUG\n" +
                            "INPUT: [1,128,128,3]\n\n" +
                            "SWAPPED FACE\n" +
                            "shape: [1,128,128,3]\n" +
                            "min: %.5f\n" +
                            "max: %.5f\n" +
                            "mean: %.5f\n\n" +
                            "DST MASK\n" +
                            "shape: [1,128,128,1]\n" +
                            "min: %.5f\n" +
                            "max: %.5f\n" +
                            "mean: %.5f\n\n" +
                            "SRC MASK\n" +
                            "shape: [1,128,128,1]\n" +
                            "min: %.5f\n" +
                            "max: %.5f\n" +
                            "mean: %.5f",
                        rgb.minOrNull() ?: 0f,
                        rgb.maxOrNull() ?: 0f,
                        rgb.average(),
                        dstMask.minOrNull() ?: 0f,
                        dstMask.maxOrNull() ?: 0f,
                        dstMask.average(),
                        srcMask.minOrNull() ?: 0f,
                        srcMask.maxOrNull() ?: 0f,
                        srcMask.average()
                    )

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

    /**
     * Reads output_2 as NHWC:
     * [1][128][128][3]
     */
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

    /**
     * Reads output_1/output_3 as NHWC:
     * [1][128][128][1]
     */
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
