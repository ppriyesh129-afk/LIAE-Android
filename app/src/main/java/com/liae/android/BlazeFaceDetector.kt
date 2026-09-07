package com.liae.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class BlazeFaceDetector(
    private val session: OrtSession
) {

    private val environment = OrtEnvironment.getEnvironment()

    companion object {
        private const val INPUT_SIZE = 128
        private const val ANCHOR_COUNT = 896
        private const val SCORE_THRESHOLD = 0.5f
        private const val IOU_THRESHOLD = 0.3f
        private const val KEYPOINT_COUNT = 6
    }

    fun detect(bitmap: Bitmap): List<BlazeFaceResult> {

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        if (originalWidth <= 0 || originalHeight <= 0) {
            return emptyList()
        }

        // ------------------------------------------------------------
        // Letterbox original image into 128 x 128.
        // ------------------------------------------------------------

        val scale =
            INPUT_SIZE.toFloat() /
                max(originalWidth, originalHeight).toFloat()

        val resizedWidth = originalWidth * scale
        val resizedHeight = originalHeight * scale

        val padX =
            (INPUT_SIZE - resizedWidth) / 2f

        val padY =
            (INPUT_SIZE - resizedHeight) / 2f

        val matrix = Matrix()

        matrix.postScale(
            scale,
            scale
        )

        matrix.postTranslate(
            padX,
            padY
        )

        val inputBitmap =
            Bitmap.createBitmap(
                INPUT_SIZE,
                INPUT_SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas = Canvas(inputBitmap)

        canvas.drawBitmap(
            bitmap,
            matrix,
            null
        )

        // ------------------------------------------------------------
        // Create BlazeFace input.
        // Model expects NCHW RGB [-1, 1].
        // ------------------------------------------------------------

        val inputBuffer =
            createInput(inputBitmap)

        val tensor =
            OnnxTensor.createTensor(
                environment,
                inputBuffer,
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

        val result =
            try {

                session.run(
                    mapOf(
                        session.inputNames.first() to tensor
                    )
                )

            } finally {

                tensor.close()
                inputBitmap.recycle()
            }

        try {

            // --------------------------------------------------------
            // Read BlazeFace outputs.
            // --------------------------------------------------------

            val regressors =
                extractFloat2D(
                    result[0].value
                )

            val scores =
                extractFloat2D(
                    result[1].value
                )

            val anchors =
                generateAnchors()

            val candidates =
                mutableListOf<Detection>()

            // --------------------------------------------------------
            // Decode 896 BlazeFace anchors.
            // --------------------------------------------------------

            val count =
                minOf(
                    ANCHOR_COUNT,
                    regressors.size,
                    scores.size,
                    anchors.size
                )

            for (i in 0 until count) {

                val regression =
                    regressors[i]

                val scoreRow =
                    scores[i]

                if (regression.size < 16) {
                    continue
                }

                if (scoreRow.isEmpty()) {
                    continue
                }

                val score =
                    sigmoid(
                        scoreRow[0]
                    )

                if (score < SCORE_THRESHOLD) {
                    continue
                }

                val anchor =
                    anchors[i]

                // ----------------------------------------------------
                // Bounding box.
                // Coordinates are normalized to 128x128 space.
                // ----------------------------------------------------

                val cx =
                    regression[0] /
                        INPUT_SIZE.toFloat() +
                        anchor[0]

                val cy =
                    regression[1] /
                        INPUT_SIZE.toFloat() +
                        anchor[1]

                val width =
                    regression[2] /
                        INPUT_SIZE.toFloat()

                val height =
                    regression[3] /
                        INPUT_SIZE.toFloat()

                val x1 =
                    cx - width / 2f

                val y1 =
                    cy - height / 2f

                val x2 =
                    cx + width / 2f

                val y2 =
                    cy + height / 2f

                // ----------------------------------------------------
                // Six BlazeFace keypoints.
                //
                // 0 = right eye
                // 1 = left eye
                // 2 = nose
                // 3 = mouth
                // 4 = right ear
                // 5 = left ear
                //
                // We preserve all six.
                // ----------------------------------------------------

                val keypoints =
                    Array(KEYPOINT_COUNT) { keypointIndex ->

                        val offset =
                            4 + keypointIndex * 2

                        floatArrayOf(
                            regression[offset] /
                                INPUT_SIZE.toFloat() +
                                anchor[0],

                            regression[offset + 1] /
                                INPUT_SIZE.toFloat() +
                                anchor[1]
                        )
                    }

                candidates.add(
                    Detection(
                        score = score,
                        x1 = x1,
                        y1 = y1,
                        x2 = x2,
                        y2 = y2,
                        keypoints = keypoints
                    )
                )
            }

            // --------------------------------------------------------
            // Weighted NMS.
            // --------------------------------------------------------

            val selected =
                weightedNms(
                    candidates
                )

            // --------------------------------------------------------
            // Convert detector coordinates from letterboxed
            // 128x128 space back into original image coordinates.
            //
            // The detector box itself is NOT forced to be square.
            // --------------------------------------------------------

            val output =
                ArrayList<BlazeFaceResult>(
                    selected.size
                )

            for (detection in selected) {

                val x1 =
                    unletterbox(
                        detection.x1,
                        padX,
                        scale,
                        originalWidth.toFloat()
                    )

                val y1 =
                    unletterbox(
                        detection.y1,
                        padY,
                        scale,
                        originalHeight.toFloat()
                    )

                val x2 =
                    unletterbox(
                        detection.x2,
                        padX,
                        scale,
                        originalWidth.toFloat()
                    )

                val y2 =
                    unletterbox(
                        detection.y2,
                        padY,
                        scale,
                        originalHeight.toFloat()
                    )

                val keypoints =
                    Array(KEYPOINT_COUNT) { k ->

                        val px =
                            unletterbox(
                                detection.keypoints[k][0],
                                padX,
                                scale,
                                originalWidth.toFloat()
                            )

                        val py =
                            unletterbox(
                                detection.keypoints[k][1],
                                padY,
                                scale,
                                originalHeight.toFloat()
                            )

                        floatArrayOf(
                            px,
                            py
                        )
                    }

                output.add(
                    BlazeFaceResult(
                        left = x1,
                        top = y1,
                        right = x2,
                        bottom = y2,
                        score = detection.score,
                        keypoints = keypoints
                    )
                )
            }

            return output

        } finally {

            result.close()
        }
    }

    // ================================================================
    // ANCHORS
    // ================================================================

    private fun generateAnchors(): List<FloatArray> {

        val anchors =
            ArrayList<FloatArray>(
                ANCHOR_COUNT
            )

        // 16 x 16 x 2 = 512
        addGridAnchors(
            anchors = anchors,
            cells = 16,
            repeats = 2
        )

        // 8 x 8 x 6 = 384
        addGridAnchors(
            anchors = anchors,
            cells = 8,
            repeats = 6
        )

        return anchors
    }

    private fun addGridAnchors(
        anchors: MutableList<FloatArray>,
        cells: Int,
        repeats: Int
    ) {

        for (y in 0 until cells) {

            for (x in 0 until cells) {

                val centerX =
                    (x + 0.5f) /
                        cells.toFloat()

                val centerY =
                    (y + 0.5f) /
                        cells.toFloat()

                repeat(repeats) {

                    anchors.add(
                        floatArrayOf(
                            centerX,
                            centerY
                        )
                    )
                }
            }
        }
    }

    // ================================================================
    // IMAGE -> BLAZEFACE TENSOR
    // ================================================================

    private fun createInput(
        bitmap: Bitmap
    ): FloatBuffer {

        val pixels =
            IntArray(
                INPUT_SIZE * INPUT_SIZE
            )

        bitmap.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val buffer =
            FloatBuffer.allocate(
                3 *
                    INPUT_SIZE *
                    INPUT_SIZE
            )

        // CHW RGB, normalized to [-1, 1].
        for (channel in 0..2) {

            for (pixel in pixels) {

                val value =
                    when (channel) {

                        0 ->
                            (pixel shr 16) and 0xFF

                        1 ->
                            (pixel shr 8) and 0xFF

                        else ->
                            pixel and 0xFF
                    }

                val normalized =
                    (
                        value.toFloat() -
                            127.5f
                        ) / 127.5f

                buffer.put(
                    normalized
                )
            }
        }

        buffer.rewind()

        return buffer
    }

    // ================================================================
    // OUTPUT PARSER
    // ================================================================

    private fun extractFloat2D(
        raw: Any
    ): Array<FloatArray> {

        return when (raw) {

            is Array<*> -> {

                if (raw.isEmpty()) {
                    emptyArray()
                } else {

                    val first =
                        raw.firstOrNull()

                    when (first) {

                        is Array<*> -> {

                            first.mapNotNull { row ->

                                when (row) {

                                    is FloatArray ->
                                        row

                                    is DoubleArray ->
                                        FloatArray(
                                            row.size
                                        ) { index ->
                                            row[index].toFloat()
                                        }

                                    else ->
                                        null
                                }

                            }.toTypedArray()
                        }

                        is FloatArray -> {

                            raw.mapNotNull { item ->
                                item as? FloatArray
                            }.toTypedArray()
                        }

                        else ->
                            emptyArray()
                    }
                }
            }

            is FloatArray ->
                arrayOf(raw)

            else ->
                emptyArray()
        }
    }

    // ================================================================
    // SIGMOID
    // ================================================================

    private fun sigmoid(
        value: Float
    ): Float {

        return (
            1.0 /
                (
                    1.0 +
                        exp(
                            -value.toDouble()
                        )
                )
            ).toFloat()
    }

    // ================================================================
    // LETTERBOX -> ORIGINAL IMAGE
    // ================================================================

    private fun unletterbox(
        value: Float,
        padding: Float,
        scale: Float,
        maximum: Float
    ): Float {

        val pixel =
            (
                value *
                    INPUT_SIZE.toFloat() -
                    padding
                ) / scale

        return pixel.coerceIn(
            0f,
            maximum
        )
    }

    // ================================================================
    // WEIGHTED NMS
    // ================================================================

    private fun weightedNms(
        detections: List<Detection>
    ): List<Detection> {

        val remaining =
            detections
                .sortedByDescending {
                    it.score
                }
                .toMutableList()

        val output =
            mutableListOf<Detection>()

        while (remaining.isNotEmpty()) {

            val top =
                remaining.removeAt(0)

            val overlapping =
                mutableListOf<Detection>()

            overlapping.add(top)

            val iterator =
                remaining.iterator()

            while (iterator.hasNext()) {

                val candidate =
                    iterator.next()

                if (
                    iou(
                        top,
                        candidate
                    ) > IOU_THRESHOLD
                ) {

                    overlapping.add(
                        candidate
                    )

                    iterator.remove()
                }
            }

            var totalWeight = 0f

            var weightedX1 = 0f
            var weightedY1 = 0f
            var weightedX2 = 0f
            var weightedY2 = 0f

            var bestScore = 0f

            val keypointSums =
                Array(KEYPOINT_COUNT) {
                    floatArrayOf(
                        0f,
                        0f
                    )
                }

            for (detection in overlapping) {

                val weight =
                    detection.score

                totalWeight += weight

                weightedX1 +=
                    detection.x1 * weight

                weightedY1 +=
                    detection.y1 * weight

                weightedX2 +=
                    detection.x2 * weight

                weightedY2 +=
                    detection.y2 * weight

                bestScore =
                    max(
                        bestScore,
                        detection.score
                    )

                for (k in 0 until KEYPOINT_COUNT) {

                    keypointSums[k][0] +=
                        detection.keypoints[k][0] *
                        weight

                    keypointSums[k][1] +=
                        detection.keypoints[k][1] *
                        weight
                }
            }

            if (totalWeight <= 0f) {
                continue
            }

            val averagedKeypoints =
                Array(KEYPOINT_COUNT) { k ->

                    floatArrayOf(
                        keypointSums[k][0] /
                            totalWeight,

                        keypointSums[k][1] /
                            totalWeight
                    )
                }

            output.add(
                Detection(
                    score = bestScore,

                    x1 =
                        weightedX1 /
                            totalWeight,

                    y1 =
                        weightedY1 /
                            totalWeight,

                    x2 =
                        weightedX2 /
                            totalWeight,

                    y2 =
                        weightedY2 /
                            totalWeight,

                    keypoints =
                        averagedKeypoints
                )
            )
        }

        return output
    }

    // ================================================================
    // IOU
    // ================================================================

    private fun iou(
        a: Detection,
        b: Detection
    ): Float {

        val left =
            max(
                a.x1,
                b.x1
            )

        val top =
            max(
                a.y1,
                b.y1
            )

        val right =
            min(
                a.x2,
                b.x2
            )

        val bottom =
            min(
                a.y2,
                b.y2
            )

        val width =
            max(
                0f,
                right - left
            )

        val height =
            max(
                0f,
                bottom - top
            )

        val intersection =
            width * height

        val areaA =
            max(
                0f,
                a.x2 - a.x1
            ) *
                max(
                    0f,
                    a.y2 - a.y1
                )

        val areaB =
            max(
                0f,
                b.x2 - b.x1
            ) *
                max(
                    0f,
                    b.y2 - b.y1
                )

        val union =
            areaA +
                areaB -
                intersection

        if (union <= 0f) {
            return 0f
        }

        return intersection / union
    }

    // ================================================================
    // INTERNAL DETECTION
    // ================================================================

    private data class Detection(
        val score: Float,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val keypoints: Array<FloatArray>
    )
}
