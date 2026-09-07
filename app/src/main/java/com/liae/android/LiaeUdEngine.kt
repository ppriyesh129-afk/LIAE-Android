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
    }

    data class Result(
        val rgb: FloatArray,
        val mask: FloatArray
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
    }

    fun run(src: FloatArray, dst: FloatArray): Result {

        val shape = longArrayOf(1, SIZE.toLong(), SIZE.toLong(), 3)

        val srcTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(src),
            shape
        )

        val dstTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(dst),
            shape
        )

        return try {
            session.run(
                mapOf(
                    "src" to srcTensor,
                    "dst" to dstTensor
                )
            ).use { outputs ->

                val rgb = flatten(outputs[0].value)
                val mask = flatten(outputs[1].value)

                Result(rgb, mask)
            }
        } finally {
            srcTensor.close()
            dstTensor.close()
        }
    }

    private fun flatten(value: Any): FloatArray {

        val list = ArrayList<Float>()

        fun walk(v: Any?) {
            when (v) {
                is FloatArray -> list.addAll(v.toList())
                is Array<*> -> v.forEach { walk(it) }
            }
        }

        walk(value)

        return list.toFloatArray()
    }

    fun close() {
        session.close()
    }
}
