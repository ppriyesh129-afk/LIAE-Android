package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint

object DflAligner {

    private const val SIZE = 128

    /*
     * DeepFaceLab FULL-face padding.
     */
    private const val DFL_PADDING = 0.2109375f

    data class AlignedFace(
        val bitmap: Bitmap,
        val forward: FloatArray,
        val inverse: FloatArray
    )

    fun align(
        source: Bitmap,
        face: BlazeFaceResult
    ): AlignedFace {

        require(source.width > 0) {
            "Invalid source bitmap width"
        }

        require(source.height > 0) {
            "Invalid source bitmap height"
        }

        require(face.keypoints.size >= 4) {
            "BlazeFace landmarks are missing"
        }

        /*
         * BlazeFace landmarks:
         *
         * 0 = right eye
         * 1 = left eye
         * 2 = nose
         * 3 = mouth
         */
        val src = arrayOf(
            Point2(face.keypoints[0][0], face.keypoints[0][1]),
            Point2(face.keypoints[1][0], face.keypoints[1][1]),
            Point2(face.keypoints[2][0], face.keypoints[2][1]),
            Point2(face.keypoints[3][0], face.keypoints[3][1])
        )

        val dst = canonicalPoints()

        // ---- DeepFaceLab-style affine estimation ----

        val srcPts = floatArrayOf(
            src[0].x, src[0].y,
            src[1].x, src[1].y,
            src[2].x, src[2].y,
            src[3].x, src[3].y
        )

        val dstPts = floatArrayOf(
            dst[0].x, dst[0].y,
            dst[1].x, dst[1].y,
            dst[2].x, dst[2].y,
            dst[3].x, dst[3].y
        )

        val matrix = Matrix()

        if (!matrix.setPolyToPoly(srcPts, 0, dstPts, 0, 4)) {
            throw IllegalStateException("Failed to estimate DFL affine transform")
        }

        val values = FloatArray(9)
        matrix.getValues(values)

        val forward = floatArrayOf(
            values[0], values[1], values[2],
            values[3], values[4], values[5]
        )

        val inverse = invertAffine(forward)

        val aligned = Bitmap.createBitmap(
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(aligned)

        val paint = Paint(
            Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
        )

        canvas.drawBitmap(source, matrix, paint)

        return AlignedFace(
            bitmap = aligned,
            forward = forward,
            inverse = inverse
        )
    }

    private fun canonicalPoints(): Array<Point2> {

        val centerX = 64f
        val centerY = 64f

        val points = arrayOf(
            Point2(89f, 43f), // right eye
            Point2(39f, 43f), // left eye
            Point2(64f, 65f), // nose
            Point2(64f, 88f)  // mouth
        )

        val paddingScale = 1f / (1f + DFL_PADDING * 2f)

        return points.map { point ->
            Point2(
                centerX + (point.x - centerX) * paddingScale,
                centerY + (point.y - centerY) * paddingScale
            )
        }.toTypedArray()
    }

    private fun invertAffine(
        matrix: FloatArray
    ): FloatArray {

        val a = matrix[0]
        val b = matrix[1]
        val tx = matrix[2]

        val c = matrix[3]
        val d = matrix[4]
        val ty = matrix[5]

        val determinant = a * d - b * c

        require(kotlin.math.abs(determinant) > 1e-8f) {
            "Non-invertible face transform"
        }

        val invA = d / determinant
        val invB = -b / determinant
        val invC = -c / determinant
        val invD = a / determinant

        val invTx = -(invA * tx + invB * ty)
        val invTy = -(invC * tx + invD * ty)

        return floatArrayOf(
            invA, invB, invTx,
            invC, invD, invTy
        )
    }

    private data class Point2(
        val x: Float,
        val y: Float
    )
}
