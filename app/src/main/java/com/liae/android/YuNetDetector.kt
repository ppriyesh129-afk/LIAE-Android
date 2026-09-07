package com.liae.android

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class YuNetDetector(
    private val context: Context
) {

    companion object {
        private const val MODEL_NAME =
            "face_detection_yunet_2023mar.onnx"

        private const val INPUT_SIZE = 640
    }

    data class Point(
        val x: Float,
        val y: Float
    )

    data class Face(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val rightEye: Point,
        val leftEye: Point,
        val nose: Point,
        val rightMouth: Point,
        val leftMouth: Point,
        val score: Float
    )

    private val environment =
        OrtEnvironment.getEnvironment()

    private val session: OrtSession

    init {

        val modelFile =
            context.getFileStreamPath(MODEL_NAME)

        if (
            !modelFile.exists() ||
            modelFile.length() < 100_000L
        ) {
            context.assets.open(MODEL_NAME).use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        require(modelFile.exists()) {
            "YuNet model was not copied"
        }

        require(modelFile.length() > 100_000L) {
            "YuNet model is too small: ${modelFile.length()}"
        }

        session =
            environment.createSession(
                modelFile.absolutePath,
                OrtSession.SessionOptions()
            )
    }

    fun detect(
        bitmap: Bitmap
    ): List<Face> {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                INPUT_SIZE,
                INPUT_SIZE,
                true
            )

        val pixels =
            IntArray(
                INPUT_SIZE * INPUT_SIZE
            )

        resized.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val planeSize =
            INPUT_SIZE * INPUT_SIZE

        val input =
            FloatArray(
                planeSize * 3
            )

        var pixelIndex = 0

        for (y in 0 until INPUT_SIZE) {

            for (x in 0 until INPUT_SIZE) {

                val pixel =
                    pixels[pixelIndex++]

                val r =
                    ((pixel shr 16) and 0xFF)
                        .toFloat()

                val g =
                    ((pixel shr 8) and 0xFF)
                        .toFloat()

                val b =
                    (pixel and 0xFF)
                        .toFloat()

                val position =
                    y * INPUT_SIZE + x

                input[position] = b

                input[
                    planeSize + position
                ] = g

                input[
                    planeSize * 2 + position
                ] = r
            }
        }

        if (resized !== bitmap) {
            resized.recycle()
        }

        val tensor =
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(input),
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

        try {

            val inputName =
                session.inputNames.first()

            session.run(
                mapOf(
                    inputName to tensor
                )
            ).use { result ->

                /*
                 * DIAGNOSTIC ONLY
                 *
                 * Do not parse the output yet.
                 */

                val outputCount =
                    result.size

                val firstOutput =
                    result[0]

                val value =
                    firstOutput.value

                val tensorInfo =
                    firstOutput.info

                val message =
                    buildString {

                        appendLine(
                            "YuNet inference succeeded."
                        )

                        appendLine(
                            "Output count: $outputCount"
                        )

                        appendLine(
                            "Output 0 type: " +
                                value.javaClass.name
                        )

                        appendLine(
                            "Output 0 info: " +
                                tensorInfo.toString()
                        )

                        when (value) {

                            is FloatArray -> {

                                appendLine(
                                    "FloatArray size: " +
                                        value.size
                                )
                            }

                            is Array<*> -> {

                                appendLine(
                                    "Array outer size: " +
                                        value.size
                                )

                                appendLine(
                                    "Array value: " +
                                        describeArray(value)
                                )
                            }
                        }
                    }

                throw IllegalStateException(
                    message
                )
            }

        } finally {

            tensor.close()
        }
    }

    private fun describeArray(
        value: Array<*>
    ): String {

        if (value.isEmpty()) {
            return "[]"
        }

        val first =
            value[0]

        return when (first) {

            is FloatArray ->
                "[${value.size}, ${first.size}]"

            is Array<*> ->
                "[${value.size}, ${describeArray(first)}]"

            else ->
                "[${value.size}, ${first?.javaClass?.name}]"
        }
    }

    fun close() {
        session.close()
    }
}
