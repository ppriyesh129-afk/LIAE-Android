package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
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

    var lastRgbShape: LongArray = longArrayOf()
        private set

    var lastMaskShape: LongArray = longArrayOf()
        private set

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

        // ONNX model input: NHWC (1,128,128,3)
        val inputShape = longArrayOf(1, SIZE.toLong(), SIZE.toLong(), 3)

        val srcTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(src),
            inputShape
        )

        val dstTensor = OnnxTensor.createTensor(
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

                val rgbTensor = outputs[0] as OnnxTensor
                val maskTensor = outputs[1] as OnnxTensor

                lastRgbShape = rgbTensor.info.shape
                lastMaskShape = maskTensor.info.shape

                Log.d("LIAE", "RGB shape = ${lastRgbShape.contentToString()}")
                Log.d("LIAE", "Mask shape = ${lastMaskShape.contentToString()}")

                Result(
                    flatten(rgbTensor.value),
                    flatten(maskTensor.value)
                )
            }

        } finally {

            srcTensor.close()
            dstTensor.close()
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
