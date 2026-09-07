package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.util.Locale

class LiaeUdEngine(context: Context) {

    companion object {
        private const val MODEL_NAME =
            "LIAE_128_80_48_16_fp32.onnx"

        private const val SIZE = 128
        private const val CHANNELS = 3
        private const val IMAGE_FLOATS =
            SIZE * SIZE * CHANNELS
        private const val MASK_FLOATS =
            SIZE * SIZE
    }

    data class Result(
        val rgb: FloatArray,
        val dstMask: FloatArray,
        val srcMask: FloatArray,
        val debugText: String
    )

    private val env =
        OrtEnvironment.getEnvironment()

    private val session: OrtSession

    init {

        val modelFile =
            context.getFileStreamPath(MODEL_NAME)

        if (!modelFile.exists()) {

            context.assets
                .open(MODEL_NAME)
                .use { input ->

                    modelFile
                        .outputStream()
                        .use { output ->

                            input.copyTo(output)
                        }
                }
        }

        session =
            env.createSession(
                modelFile.absolutePath,
                OrtSession.SessionOptions()
            )

        /*
         * Verify the actual model interface at startup.
         */
        val inputs = session.inputNames

        require(inputs.size == 1) {
            "Expected 1 ONNX input, got ${inputs.size}"
        }

        require(inputs.contains("in_face")) {
            "Expected ONNX input 'in_face', got $inputs"
        }

        val outputs = session.outputNames

        require(outputs.size == 3) {
            "Expected 3 ONNX outputs, got ${outputs.size}"
        }
    }

    fun run(
        dst: FloatArray
    ): Result {

        require(
            dst.size == IMAGE_FLOATS
        ) {
            "Target tensor size must be " +
                "$IMAGE_FLOATS, got ${dst.size}"
        }

        val inputShape =
            longArrayOf(
                1,
                SIZE.toLong(),
                SIZE.toLong(),
                CHANNELS.toLong()
            )

        val dstTensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(dst),
                inputShape
            )

        return try {

            session.run(
                mapOf(
                    "in_face" to dstTensor
                )
            ).use { outputs ->

                /*
                 * Verified ONNX output order:
                 *
                 * output_1 = destination mask
                 * output_2 = swapped face
                 * output_3 = source/swapped mask
                 */

                val dstMaskTensor =
                    outputs["output_1"]
                        ?.get() as? OnnxTensor
                        ?: throw IllegalStateException(
                            "Missing output_1 (destination mask)"
                        )

                val rgbTensor =
                    outputs["output_2"]
                        ?.get() as? OnnxTensor
                        ?: throw IllegalStateException(
                            "Missing output_2 (swapped face)"
                        )

                val srcMaskTensor =
                    outputs["output_3"]
                        ?.get() as? OnnxTensor
                        ?: throw IllegalStateException(
                            "Missing output_3 (source mask)"
                        )

                val rgb =
                    flatten(
                        rgbTensor.value
                    )

                val dstMask =
                    flatten(
                        dstMaskTensor.value
                    )

                val srcMask =
                    flatten(
                        srcMaskTensor.value
                    )

                require(
                    rgb.size == IMAGE_FLOATS
                ) {
                    "Unexpected RGB size: ${rgb.size}"
                }

                require(
                    dstMask.size == MASK_FLOATS
                ) {
                    "Unexpected destination mask size: " +
                        dstMask.size
                }

                require(
                    srcMask.size == MASK_FLOATS
                ) {
                    "Unexpected source mask size: " +
                        srcMask.size
                }

                val rgbMin =
                    rgb.minOrNull() ?: 0f

                val rgbMax =
                    rgb.maxOrNull() ?: 0f

                val rgbMean =
                    rgb.average()

                val dstMaskMin =
                    dstMask.minOrNull() ?: 0f

                val dstMaskMax =
                    dstMask.maxOrNull() ?: 0f

                val dstMaskMean =
                    dstMask.average()

                val srcMaskMin =
                    srcMask.minOrNull() ?: 0f

                val srcMaskMax =
                    srcMask.maxOrNull() ?: 0f

                val srcMaskMean =
                    srcMask.average()

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
                        rgbMin,
                        rgbMax,
                        rgbMean,
                        dstMaskMin,
                        dstMaskMax,
                        dstMaskMean,
                        srcMaskMin,
                        srcMaskMax,
                        srcMaskMean
                    )

                Result(
                    rgb = rgb,
                    dstMask = dstMask,
                    srcMask = srcMask,
                    debugText = debugText
                )
            }

        } finally {

            dstTensor.close()
        }
    }

    private fun flatten(
        value: Any
    ): FloatArray {

        val out =
            ArrayList<Float>()

        fun walk(
            v: Any?
        ) {

            when (v) {

                is FloatArray -> {

                    for (f in v) {
                        out.add(f)
                    }
                }

                is Array<*> -> {

                    for (child in v) {
                        walk(child)
                    }
                }
            }
        }

        walk(value)

        return out.toFloatArray()
    }

    fun close() {
        session.close()
    }
}
