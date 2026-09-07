package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

class LiaeUdEngine(
    context: Context
) {

    companion object {

        private const val MODEL_NAME =
            "LIAE_128_80_48_16_fp32.onnx"

        private const val SIZE = 128
    }

    data class Result(
        val rgb: FloatArray,
        val mask: FloatArray
    )

    private val env =
        OrtEnvironment.getEnvironment()

    private val session: OrtSession

    var lastRgbShape: LongArray =
        longArrayOf()
        private set

    var lastMaskShape: LongArray =
        longArrayOf()
        private set

    init {

        val modelFile =
            context.getFileStreamPath(
                MODEL_NAME
            )

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
            src.size ==
                SIZE * SIZE * 3
        ) {
            "Source tensor size must be " +
                "${SIZE * SIZE * 3}, got ${src.size}"
        }

        require(
            dst.size ==
                SIZE * SIZE * 3
        ) {
            "Target tensor size must be " +
                "${SIZE * SIZE * 3}, got ${dst.size}"
        }

        /*
         * VERIFIED MODEL INPUT
         *
         * [1, 128, 128, 3]
         *
         * NHWC
         */
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
                        .orElseThrow {
                            IllegalStateException(
                                "Missing output_1"
                            )
                        } as OnnxTensor

                val maskTensor =
                    outputs["output_2"]
                        .orElseThrow {
                            IllegalStateException(
                                "Missing output_2"
                            )
                        } as OnnxTensor

                lastRgbShape =
                    rgbTensor.info.shape

                lastMaskShape =
                    maskTensor.info.shape

                Log.d(
                    "LIAE",
                    "RGB shape = " +
                        lastRgbShape
                            .contentToString()
                )

                Log.d(
                    "LIAE",
                    "Mask shape = " +
                        lastMaskShape
                            .contentToString()
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
                    rgb.size ==
                        SIZE * SIZE * 3
                ) {
                    "Unexpected RGB size: ${rgb.size}"
                }

                require(
                    mask.size ==
                        SIZE * SIZE
                ) {
                    "Unexpected Mask size: ${mask.size}"
                }

                Result(
                    rgb = rgb,
                    mask = mask
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
