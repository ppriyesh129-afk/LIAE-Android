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

        // Verify exported ONNX interface.
        require(session.inputNames == setOf("in_face")) {
            "Expected ONNX input 'in_face', got ${session.inputNames}"
        }

        require(session.outputNames.containsAll(setOf("output_1", "output_2", "output_3"))) {
            "Unexpected ONNX outputs: ${session.outputNames}"
        }
    }

    fun run(dst: FloatArray): Result {

        require(dst.size == IMAGE_FLOATS) {
            "Target tensor size must be $IMAGE_FLOATS, got ${dst.size}"
        }

        val inputShape = longArrayOf(
            1,
            SIZE.toLong(),
            SIZE.toLong(),
            CHANNELS.toLong()
        )

        val inFaceTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(dst),
            inputShape
        )

        return try {

            session.run(
                mapOf("in_face" to inFaceTensor)
            ).use { outputs ->

                // Verified output order:
                // output_1 = destination mask
                // output_2 = swapped face
                // output_3 = source mask

                val dstMaskTensor =
                    outputs["output_1"]?.get() as? OnnxTensor
                        ?: error("Missing output_1")

                val rgbTensor =
                    outputs["output_2"]?.get() as? OnnxTensor
                        ?: error("Missing output_2")

                val srcMaskTensor =
                    outputs["output_3"]?.get() as? OnnxTensor
                        ?: error("Missing output_3")

                val rgb = flatten(rgbTensor.value)
                val dstMask = flatten(dstMaskTensor.value)
                val srcMask = flatten(srcMaskTensor.value)

                require(rgb.size == IMAGE_FLOATS)
                require(dstMask.size == MASK_FLOATS)
                require(srcMask.size == MASK_FLOATS)

                val debugText = String.format(
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
            inFaceTensor.close()
        }
    }

    private fun flatten(value: Any): FloatArray {

        val out = ArrayList<Float>()

        fun walk(v: Any?) {
            when (v) {
                is FloatArray -> out.addAll(v.toList())
                is Array<*> -> v.forEach { walk(it) }
            }
        }

        walk(value)
        return out.toFloatArray()
    }

    fun close() {
        session.close()
    }
}
