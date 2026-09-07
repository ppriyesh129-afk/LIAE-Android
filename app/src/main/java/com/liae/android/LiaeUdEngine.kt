package com.liae.android

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class LiaeUdEngine(context: Context) {

    companion object {
        private const val MODEL_NAME = "LIAE_128_80_48_16_fp32.onnx"
        private const val SIZE = 128
        private const val CHANNELS = 3
    }

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(MODEL_NAME).use {
            it.readBytes()
        }

        val options = OrtSession.SessionOptions()

        session = environment.createSession(
            modelBytes,
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

        require(src.size == SIZE * SIZE * CHANNELS) {
            "src must contain 128x128x3 floats"
        }

        require(dst.size == SIZE * SIZE * CHANNELS) {
            "dst must contain 128x128x3 floats"
        }

        val srcTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(src),
            longArrayOf(1, SIZE.toLong(), SIZE.toLong(), CHANNELS.toLong())
        )

        val dstTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(dst),
            longArrayOf(1, SIZE.toLong(), SIZE.toLong(), CHANNELS.toLong())
        )

        srcTensor.use { srcT ->
            dstTensor.use { dstT ->

                val inputs = mapOf(
                    "src" to srcT,
                    "dst" to dstT
                )

                session.run(inputs).use { output ->

                    val rgb = extractFloatArray(output[0].value)
                    val mask = extractFloatArray(output[1].value)

                    return Result(
                        rgb = rgb,
                        mask = mask
                    )
                }
            }
        }
    }

    private fun extractFloatArray(value: Any): FloatArray {

        @Suppress("UNCHECKED_CAST")
        val tensor = value as Array<Array<Array<FloatArray>>>

        val height = tensor.size
        val width = tensor[0].size
        val channels = tensor[0][0].size

        val result = FloatArray(
            height * width * channels
        )

        var index = 0

        for (y in 0 until height) {
            for (x in 0 until width) {
                for (c in 0 until channels) {
                    result[index++] = tensor[y][x][c]
                }
            }
        }

        return result
    }

    fun close() {
        session.close()
        environment.close()
    }
}
