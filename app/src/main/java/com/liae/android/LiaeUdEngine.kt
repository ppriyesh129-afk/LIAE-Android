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
    }

    data class Result(
        val rgb: FloatArray,
        val mask: FloatArray,
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
    }

    fun run(
        src: FloatArray,
        dst: FloatArray
    ): Result {

        require(
            src.size == SIZE * SIZE * 3
        ) {
            "Source tensor size must be " +
                "${SIZE * SIZE * 3}, got ${src.size}"
        }

        require(
            dst.size == SIZE * SIZE * 3
        ) {
            "Target tensor size must be " +
                "${SIZE * SIZE * 3}, got ${dst.size}"
        }

        val inputShape =
            longArrayOf(
                1,
                SIZE.toLong(),
                SIZE.toLong(),
                3
            )

        val srcTensor =
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(src),
                inputShape
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
                    "src" to srcTensor,
                    "dst" to dstTensor
                )
            ).use { outputs ->

                val rgbTensor =
                    outputs["output_1"]
                        ?.get() as? OnnxTensor
                        ?: throw IllegalStateException(
                            "Missing output_1"
                        )

                val maskTensor =
                    outputs["output_2"]
                        ?.get() as? OnnxTensor
                        ?: throw IllegalStateException(
                            "Missing output_2"
                        )

                val rgb =
                    flatten(
                        rgbTensor.value
                    )

                val mask =
                    flatten(
                        maskTensor.value
                    )

                require(
                    rgb.size == SIZE * SIZE * 3
                ) {
                    "Unexpected RGB size: ${rgb.size}"
                }

                require(
                    mask.size == SIZE * SIZE
                ) {
                    "Unexpected Mask size: ${mask.size}"
                }

                val rgbMin =
                    rgb.minOrNull() ?: 0f

                val rgbMax =
                    rgb.maxOrNull() ?: 0f

                val rgbMean =
                    rgb.average()

                val maskMin =
                    mask.minOrNull() ?: 0f

                val maskMax =
                    mask.maxOrNull() ?: 0f

                val maskMean =
                    mask.average()

                val debugText =
                    String.format(
                        Locale.US,
                        "LIAE DEBUG\n" +
                            "RGB shape: [1,128,128,3]\n" +
                            "MASK shape: [1,128,128,1]\n\n" +
                            "RGB min: %.5f\n" +
                            "RGB max: %.5f\n" +
                            "RGB mean: %.5f\n\n" +
                            "MASK min: %.5f\n" +
                            "MASK max: %.5f\n" +
                            "MASK mean: %.5f",
                        rgbMin,
                        rgbMax,
                        rgbMean,
                        maskMin,
                        maskMax,
                        maskMean
                    )

                Result(
                    rgb = rgb,
                    mask = mask,
                    debugText = debugText
                )
            }

        } finally {

            srcTensor.close()
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
