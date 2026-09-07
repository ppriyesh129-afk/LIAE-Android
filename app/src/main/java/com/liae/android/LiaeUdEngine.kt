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

        /*
         * Copy the ONNX model from APK assets to the app's
         * internal files directory.
         *
         * This avoids:
         *
         *     assets -> readBytes() -> 158 MB Java ByteArray
         *
         * Instead ONNX Runtime opens the actual file directly.
         */

        val modelFile = context.getFileStreamPath(MODEL_NAME)

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
        val mask: FloatArray
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

                return Result(
                    rgb = rgb,
                    mask = mask
                )
            }

        } finally {

            srcTensor.close()
            dstTensor.close()
        }
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

                FloatArray(values.size) { index ->
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
