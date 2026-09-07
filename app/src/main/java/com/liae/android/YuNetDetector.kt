package com.liae.android

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YuNetDetector(
    private val context: Context
) {

    companion object {

        private const val MODEL_NAME =
            "face_detection_yunet_2023mar.onnx"

        /*
         * This YuNet ONNX expects:
         *
         * [1, 3, 640, 640]
         *
         * NCHW
         */

        private const val INPUT_SIZE = 640

        private const val SCORE_THRESHOLD = 0.6f
        private const val NMS_THRESHOLD = 0.3f
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
            context.getFileStreamPath(
                MODEL_NAME
            )

        if (
            !modelFile.exists() ||
            modelFile.length() < 100_000L
        ) {

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

        require(
            modelFile.exists()
        ) {
            "YuNet model was not copied"
        }

        require(
            modelFile.length() > 100_000L
        ) {
            "YuNet model is too small: " +
                modelFile.length()
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

        val originalWidth =
            bitmap.width.toFloat()

        val originalHeight =
            bitmap.height.toFloat()

        /*
         * Resize image to 640x640.
         */
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

        /*
         * NCHW input:
         *
         * channel 0 = B
         * channel 1 = G
         * channel 2 = R
         *
         * Layout:
         *
         * [BBBB...][GGGG...][RRRR...]
         */

        val planeSize =
            INPUT_SIZE * INPUT_SIZE

        val input =
            FloatArray(
                planeSize * 3
            )

        var pixelIndex = 0

        for (
            y in 0 until INPUT_SIZE
        ) {

            for (
                x in 0 until INPUT_SIZE
            ) {

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

                input[position] =
                    b

                input[
                    planeSize + position
                ] =
                    g

                input[
                    planeSize * 2 + position
                ] =
                    r
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
                 * Positional output access.
                 */
                val raw =
                    result[0].value

                val values =
                    flatten(raw)

                return parseOutput(
                    values,
                    originalWidth,
                    originalHeight
                )
            }

        } finally {

            tensor.close()
        }
    }

    private fun parseOutput(
        values: FloatArray,
        originalWidth: Float,
        originalHeight: Float
    ): List<Face> {

        /*
         * YuNet detection format:
         *
         * x
         * y
         * width
         * height
         * right eye x/y
         * left eye x/y
         * nose x/y
         * right mouth x/y
         * left mouth x/y
         * score
         */

        val stride = 15

        if (
            values.size % stride != 0
        ) {

            throw IllegalStateException(
                "Unexpected YuNet output size: " +
                    values.size +
                    " (not divisible by 15)"
            )
        }

        val scaleX =
            originalWidth /
                INPUT_SIZE.toFloat()

        val scaleY =
            originalHeight /
                INPUT_SIZE.toFloat()

        val detections =
            ArrayList<Face>()

        var offset = 0

        while (
            offset + stride <= values.size
        ) {

            val score =
                values[offset + 14]

            if (
                score >= SCORE_THRESHOLD
            ) {

                val x =
                    values[offset] *
                        scaleX

                val y =
                    values[offset + 1] *
                        scaleY

                val width =
                    values[offset + 2] *
                        scaleX

                val height =
                    values[offset + 3] *
                        scaleY

                val rightEye =
                    Point(
                        values[offset + 4] *
                            scaleX,
                        values[offset + 5] *
                            scaleY
                    )

                val leftEye =
                    Point(
                        values[offset + 6] *
                            scaleX,
                        values[offset + 7] *
                            scaleY
                    )

                val nose =
                    Point(
                        values[offset + 8] *
                            scaleX,
                        values[offset + 9] *
                            scaleY
                    )

                val rightMouth =
                    Point(
                        values[offset + 10] *
                            scaleX,
                        values[offset + 11] *
                            scaleY
                    )

                val leftMouth =
                    Point(
                        values[offset + 12] *
                            scaleX,
                        values[offset + 13] *
                            scaleY
                    )

                detections.add(
                    Face(
                        x = x,
                        y = y,
                        width = width,
                        height = height,
                        rightEye = rightEye,
                        leftEye = leftEye,
                        nose = nose,
                        rightMouth = rightMouth,
                        leftMouth = leftMouth,
                        score = score
                    )
                )
            }

            offset += stride
        }

        return nms(detections)
    }

    private fun nms(
        faces: List<Face>
    ): List<Face> {

        if (faces.isEmpty()) {
            return emptyList()
        }

        val sorted =
            faces
                .sortedByDescending {
                    it.score
                }
                .toMutableList()

        val selected =
            ArrayList<Face>()

        while (
            sorted.isNotEmpty()
        ) {

            val best =
                sorted.removeAt(0)

            selected.add(best)

            val iterator =
                sorted.iterator()

            while (
                iterator.hasNext()
            ) {

                val candidate =
                    iterator.next()

                if (
                    iou(
                        best,
                        candidate
                    ) > NMS_THRESHOLD
                ) {

                    iterator.remove()
                }
            }
        }

        return selected
    }

    private fun iou(
        a: Face,
        b: Face
    ): Float {

        val left =
            max(
                a.x,
                b.x
            )

        val top =
            max(
                a.y,
                b.y
            )

        val right =
            min(
                a.x + a.width,
                b.x + b.width
            )

        val bottom =
            min(
                a.y + a.height,
                b.y + b.height
            )

        val intersectionWidth =
            max(
                0f,
                right - left
            )

        val intersectionHeight =
            max(
                0f,
                bottom - top
            )

        val intersection =
            intersectionWidth *
                intersectionHeight

        val areaA =
            max(
                0f,
                a.width
            ) *
                max(
                    0f,
                    a.height
                )

        val areaB =
            max(
                0f,
                b.width
            ) *
                max(
                    0f,
                    b.height
                )

        val union =
            areaA +
                areaB -
                intersection

        if (
            union <= 0f
        ) {
            return 0f
        }

        return intersection / union
    }

    private fun flatten(
        value: Any
    ): FloatArray {

        return when (value) {

            is FloatArray -> {
                value.copyOf()
            }

            is Array<*> -> {

                val list =
                    ArrayList<Float>()

                fun collect(
                    item: Any?
                ) {

                    when (item) {

                        is FloatArray -> {

                            for (v in item) {
                                list.add(v)
                            }
                        }

                        is Array<*> -> {

                            for (
                                child in item
                            ) {
                                collect(child)
                            }
                        }
                    }
                }

                collect(value)

                FloatArray(
                    list.size
                ) {
                    list[it]
                }
            }

            else -> {

                throw IllegalStateException(
                    "Unsupported YuNet output type: " +
                        value.javaClass.name
                )
            }
        }
    }

    fun close() {
        session.close()
    }
}
