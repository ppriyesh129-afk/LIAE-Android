package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint

object DflAligner {

    private const val SIZE = 128
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

        require(source.width > 0)
        require(source.height > 0)
        require(face.keypoints.size >= 4)

        val srcPts = floatArrayOf(
            face.keypoints[0][0], face.keypoints[0][1], // right eye
            face.keypoints[1][0], face.keypoints[1][1], // left eye
            face.keypoints[2][0], face.keypoints[2][1], // nose
            face.keypoints[3][0], face.keypoints[3][1]  // mouth
        )

        val dst = canonicalPoints()

        val dstPts = floatArrayOf(
            dst[0].x, dst[0].y,
            dst[1].x, dst[1].y,
            dst[2].x, dst[2].y,
            dst[3].x, dst[3].y
        )

        val matrix = Matrix()

        // IMPORTANT:
        // Use only THREE point pairs.
        // 3 = affine transform (6 parameters)
        // 4 = perspective transform (9 parameters)
        if (!matrix.setPolyToPoly(srcPts, 0, dstPts, 0, 3)) {
            throw IllegalStateException("Failed to estimate affine transform")
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

        paint.isFilterBitmap = true

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

        val scale = 1f / (1f + DFL_PADDING * 2f)

        val base = arrayOf(
            Point2(89f, 43f), // right eye
            Point2(39f, 43f), // left eye
            Point2(64f, 65f), // nose
            Point2(64f, 88f)  // mouth
        )

        return base.map {
            Point2(
                centerX + (it.x - centerX) * scale,
                centerY + (it.y - centerY) * scale
            )
        }.toTypedArray()
    }

    private fun invertAffine(m: FloatArray): FloatArray {

        val a = m[0]
        val b = m[1]
        val tx = m[2]

        val c = m[3]
        val d = m[4]
        val ty = m[5]

        val det = a * d - b * c

        require(kotlin.math.abs(det) > 1e-8f) {
            "Non-invertible transform"
        }

        val ia = d / det
        val ib = -b / det
        val ic = -c / det
        val id = a / det

        val itx = -(ia * tx + ib * ty)
        val ity = -(ic * tx + id * ty)

        return floatArrayOf(
            ia, ib, itx,
            ic, id, ity
        )
    }

    private data class Point2(
        val x: Float,
        val y: Float
    )
}
