package com.liae.android

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class YuNetDetector(
    private val context: Context
) {

    companion object {

        private const val MODEL_NAME =
            "face_detection_yunet_2023mar.onnx"

        private const val INPUT_SIZE = 640

        private const val SCORE_THRESHOLD = 0.60f

        private const val NMS_THRESHOLD = 0.30f

        private const val TOP_K = 5000

        private val STRIDES =
            intArrayOf(8, 16, 32)
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

    private data class Detection(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val landmarks: FloatArray,
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

        require(modelFile.exists()) {
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

        require(bitmap.width > 0) {
            "Invalid bitmap width"
        }

        require(bitmap.height > 0) {
            "Invalid bitmap height"
        }

        /*
         * The 2023mar ONNX model has fixed
         * 640x640 input dimensions.
         *
         * OpenCV's blobFromImage uses:
         *
         * BGR
         * 0..255
         * NCHW
         *
         * No normalization.
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
                    (
                        pixel shr 16
                    ) and 0xFF

                val g =
                    (
                        pixel shr 8
                    ) and 0xFF

                val b =
                    pixel and 0xFF

                val position =
                    y * INPUT_SIZE + x

                /*
                 * B channel
                 */
                input[position] =
                    b.toFloat()

                /*
                 * G channel
                 */
                input[
                    planeSize + position
                ] =
                    g.toFloat()

                /*
                 * R channel
                 */
                input[
                    planeSize * 2 + position
                ] =
                    r.toFloat()
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

                require(result.size() == 12) {
                    "YuNet expected 12 outputs, got " +
                        result.size()
                }

                /*
                 * IMPORTANT:
                 *
                 * Actual OpenCV YuNet output order:
                 *
                 * 0  cls_8
                 * 1  cls_16
                 * 2  cls_32
                 *
                 * 3  obj_8
                 * 4  obj_16
                 * 5  obj_32
                 *
                 * 6  bbox_8
                 * 7  bbox_16
                 * 8  bbox_32
                 *
                 * 9  kps_8
                 * 10 kps_16
                 * 11 kps_32
                 */

                val cls8 =
                    flattenTensor(
                        result[0].value
                    )

                val cls16 =
                    flattenTensor(
                        result[1].value
                    )

                val cls32 =
                    flattenTensor(
                        result[2].value
                    )

                val obj8 =
                    flattenTensor(
                        result[3].value
                    )

                val obj16 =
                    flattenTensor(
                        result[4].value
                    )

                val obj32 =
                    flattenTensor(
                        result[5].value
                    )

                val bbox8 =
                    flattenTensor(
                        result[6].value
                    )

                val bbox16 =
                    flattenTensor(
                        result[7].value
                    )

                val bbox32 =
                    flattenTensor(
                        result[8].value
                    )

                val kps8 =
                    flattenTensor(
                        result[9].value
                    )

                val kps16 =
                    flattenTensor(
                        result[10].value
                    )

                val kps32 =
                    flattenTensor(
                        result[11].value
                    )

                val detections =
                    ArrayList<Detection>()

                decodeStride(
                    stride = 8,
                    cls = cls8,
                    obj = obj8,
                    bbox = bbox8,
                    kps = kps8,
                    detections = detections
                )

                decodeStride(
                    stride = 16,
                    cls = cls16,
                    obj = obj16,
                    bbox = bbox16,
                    kps = kps16,
                    detections = detections
                )

                decodeStride(
                    stride = 32,
                    cls = cls32,
                    obj = obj32,
                    bbox = bbox32,
                    kps = kps32,
                    detections = detections
                )

                val candidates =
                    detections
                        .sortedByDescending {
                            it.score
                        }
                        .take(TOP_K)

                val finalDetections =
                    nms(
                        candidates,
                        NMS_THRESHOLD
                    )

                /*
                 * Convert from 640x640 detector
                 * coordinates back to original image.
                 */

                val scaleX =
                    bitmap.width.toFloat() /
                        INPUT_SIZE.toFloat()

                val scaleY =
                    bitmap.height.toFloat() /
                        INPUT_SIZE.toFloat()

                return finalDetections.map { d ->

                    Face(

                        x =
                            d.x * scaleX,

                        y =
                            d.y * scaleY,

                        width =
                            d.width * scaleX,

                        height =
                            d.height * scaleY,

                        rightEye =
                            Point(
                                d.landmarks[0] *
                                    scaleX,

                                d.landmarks[1] *
                                    scaleY
                            ),

                        leftEye =
                            Point(
                                d.landmarks[2] *
                                    scaleX,

                                d.landmarks[3] *
                                    scaleY
                            ),

                        nose =
                            Point(
                                d.landmarks[4] *
                                    scaleX,

                                d.landmarks[5] *
                                    scaleY
                            ),

                        rightMouth =
                            Point(
                                d.landmarks[6] *
                                    scaleX,

                                d.landmarks[7] *
                                    scaleY
                            ),

                        leftMouth =
                            Point(
                                d.landmarks[8] *
                                    scaleX,

                                d.landmarks[9] *
                                    scaleY
                            ),

                        score =
                            d.score
                    )
                }
            }

        } finally {

            tensor.close()
        }
    }

    private fun decodeStride(
        stride: Int,
        cls: FloatArray,
        obj: FloatArray,
        bbox: FloatArray,
        kps: FloatArray,
        detections: MutableList<Detection>
    ) {

        val cols =
            INPUT_SIZE / stride

        val rows =
            INPUT_SIZE / stride

        val count =
            rows * cols

        require(
            cls.size == count
        ) {
            "YuNet cls_$stride expected " +
                count +
                " values, got " +
                cls.size
        }

        require(
            obj.size == count
        ) {
            "YuNet obj_$stride expected " +
                count +
                " values, got " +
                obj.size
        }

        require(
            bbox.size == count * 4
        ) {
            "YuNet bbox_$stride expected " +
                count * 4 +
                " values, got " +
                bbox.size
        }

        require(
            kps.size == count * 10
        ) {
            "YuNet kps_$stride expected " +
                count * 10 +
                " values, got " +
                kps.size
        }

        for (r in 0 until rows) {

            for (c in 0 until cols) {

                val index =
                    r * cols + c

                /*
                 * OpenCV implementation:
                 *
                 * cls_score = clamp(cls)
                 * obj_score = clamp(obj)
                 * score =
                 * sqrt(cls_score * obj_score)
                 */

                val clsScore =
                    cls[index]
                        .coerceIn(
                            0.0f,
                            1.0f
                        )

                val objScore =
                    obj[index]
                        .coerceIn(
                            0.0f,
                            1.0f
                        )

                val score =
                    sqrt(
                        (
                            clsScore *
                                objScore
                        ).toDouble()
                    ).toFloat()

                if (
                    score <
                    SCORE_THRESHOLD
                ) {
                    continue
                }

                /*
                 * Bounding box.
                 *
                 * OpenCV:
                 *
                 * cx =
                 * (c + bbox[0]) * stride
                 *
                 * cy =
                 * (r + bbox[1]) * stride
                 *
                 * w =
                 * exp(bbox[2]) * stride
                 *
                 * h =
                 * exp(bbox[3]) * stride
                 */

                val bboxIndex =
                    index * 4

                val cx =
                    (
                        c +
                            bbox[
                                bboxIndex
                            ]
                    ) *
                        stride.toFloat()

                val cy =
                    (
                        r +
                            bbox[
                                bboxIndex + 1
                            ]
                    ) *
                        stride.toFloat()

                val width =
                    exp(
                        bbox[
                            bboxIndex + 2
                        ].toDouble()
                    ).toFloat() *
                        stride.toFloat()

                val height =
                    exp(
                        bbox[
                            bboxIndex + 3
                        ].toDouble()
                    ).toFloat() *
                        stride.toFloat()

                val x =
                    cx -
                        width / 2.0f

                val y =
                    cy -
                        height / 2.0f

                /*
                 * Five landmarks:
                 *
                 * right eye
                 * left eye
                 * nose
                 * right mouth
                 * left mouth
                 */

                val landmarks =
                    FloatArray(10)

                val kpsIndex =
                    index * 10

                for (n in 0 until 5) {

                    landmarks[
                        n * 2
                    ] =
                        (
                            kps[
                                kpsIndex +
                                    n * 2
                            ] +
                                c
                        ) *
                            stride.toFloat()

                    landmarks[
                        n * 2 + 1
                    ] =
                        (
                            kps[
                                kpsIndex +
                                    n * 2 +
                                    1
                            ] +
                                r
                        ) *
                            stride.toFloat()
                }

                detections.add(
                    Detection(
                        x = x,
                        y = y,
                        width = width,
                        height = height,
                        landmarks = landmarks,
                        score = score
                    )
                )
            }
        }
    }

    private fun nms(
        detections: List<Detection>,
        threshold: Float
    ): List<Detection> {

        if (detections.isEmpty()) {
            return emptyList()
        }

        val remaining =
            detections
                .sortedByDescending {
                    it.score
                }
                .toMutableList()

        val selected =
            ArrayList<Detection>()

        while (
            remaining.isNotEmpty()
        ) {

            val best =
                remaining.removeAt(0)

            selected.add(best)

            val iterator =
                remaining.iterator()

            while (
                iterator.hasNext()
            ) {

                val candidate =
                    iterator.next()

                if (
                    intersectionOverUnion(
                        best,
                        candidate
                    ) >= threshold
                ) {

                    iterator.remove()
                }
            }
        }

        return selected
    }

    private fun intersectionOverUnion(
        a: Detection,
        b: Detection
    ): Float {

        val ax1 =
            a.x

        val ay1 =
            a.y

        val ax2 =
            a.x + a.width

        val ay2 =
            a.y + a.height

        val bx1 =
            b.x

        val by1 =
            b.y

        val bx2 =
            b.x + b.width

        val by2 =
            b.y + b.height

        val ix1 =
            max(
                ax1,
                bx1
            )

        val iy1 =
            max(
                ay1,
                by1
            )

        val ix2 =
            min(
                ax2,
                bx2
            )

        val iy2 =
            min(
                ay2,
                by2
            )

        val intersectionWidth =
            max(
                0.0f,
                ix2 - ix1
            )

        val intersectionHeight =
            max(
                0.0f,
                iy2 - iy1
            )

        val intersection =
            intersectionWidth *
                intersectionHeight

        if (
            intersection <= 0.0f
        ) {
            return 0.0f
        }

        val areaA =
            max(
                0.0f,
                a.width
            ) *
                max(
                    0.0f,
                    a.height
                )

        val areaB =
            max(
                0.0f,
                b.width
            ) *
                max(
                    0.0f,
                    b.height
                )

        val union =
            areaA +
                areaB -
                intersection

        if (
            union <= 0.0f
        ) {
            return 0.0f
        }

        return intersection / union
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

                fun collect(
                    item: Any?
                ) {

                    when (item) {

                        is FloatArray -> {

                            for (
                                v in item
                            ) {
                                values.add(v)
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
                    values.size
                ) { index ->
                    values[index]
                }
            }

            else -> {

                throw IllegalArgumentException(
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
