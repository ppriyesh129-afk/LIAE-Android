package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

object DflAligner {

    private const val SIZE = 128

    // DeepFaceLab FULL-face padding
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

        require(source.width > 0 && source.height > 0)

        val w = face.right - face.left
        val h = face.bottom - face.top

        val cx = (face.left + face.right) * 0.5f
        val cy = (face.top + face.bottom) * 0.5f

        // Approximate DeepFaceLab five-point template
        val src = arrayOf(
            Point2(cx - w * 0.18f, cy - h * 0.15f), // right eye
            Point2(cx + w * 0.18f, cy - h * 0.15f), // left eye
            Point2(cx,             cy + h * 0.02f), // nose
            Point2(cx - w * 0.12f, cy + h * 0.22f), // right mouth
            Point2(cx + w * 0.12f, cy + h * 0.22f)  // left mouth
        )

        val dst = canonicalPoints()

        val transform = estimateSimilarityTransform(src, dst)
        val inverse = invertAffine(transform)

        val matrix = Matrix().apply {
            setValues(
                floatArrayOf(
                    transform[0], transform[1], transform[2],
                    transform[3], transform[4], transform[5],
                    0f, 0f, 1f
                )
            )
        }

        val aligned = Bitmap.createBitmap(
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )

        Canvas(aligned).drawBitmap(
            source,
            matrix,
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        )

        return AlignedFace(aligned, transform, inverse)
    }

    private fun canonicalPoints(): Array<Point2> {

        val eyeY = 43f
        val eyeLX = 39f
        val eyeRX = 89f

        val noseX = 64f
        val noseY = 65f

        val mouthLX = 45f
        val mouthRX = 83f
        val mouthY = 88f

        val cx = 64f
        val cy = 64f

        val scale = 1f / (1f + DFL_PADDING * 2f)

        fun expand(x: Float, y: Float): Point2 =
            Point2(
                cx + (x - cx) * scale,
                cy + (y - cy) * scale
            )

        return arrayOf(
            expand(eyeLX, eyeY),
            expand(eyeRX, eyeY),
            expand(noseX, noseY),
            expand(mouthLX, mouthY),
            expand(mouthRX, mouthY)
        )
    }

    private fun estimateSimilarityTransform(
        src: Array<Point2>,
        dst: Array<Point2>
    ): FloatArray {

        var srcCx = 0.0
        var srcCy = 0.0
        var dstCx = 0.0
        var dstCy = 0.0

        for (i in 0 until 5) {
            srcCx += src[i].x
            srcCy += src[i].y
            dstCx += dst[i].x
            dstCy += dst[i].y
        }

        srcCx /= 5.0
        srcCy /= 5.0
        dstCx /= 5.0
        dstCy /= 5.0

        var a = 0.0
        var b = 0.0
        var denom = 0.0

        for (i in 0 until 5) {

            val sx = src[i].x - srcCx
            val sy = src[i].y - srcCy

            val dx = dst[i].x - dstCx
            val dy = dst[i].y - dstCy

            a += sx * dx + sy * dy
            b += sx * dy - sy * dx
            denom += sx * sx + sy * sy
        }

        val scale = sqrt(a * a + b * b) / denom
        val angle = atan2(b, a)

        val c = cos(angle) * scale
        val s = sin(angle) * scale

        val tx = dstCx - (c * srcCx - s * srcCy)
        val ty = dstCy - (s * srcCx + c * srcCy)

        return floatArrayOf(
            c.toFloat(),
            (-s).toFloat(),
            tx.toFloat(),
            s.toFloat(),
            c.toFloat(),
            ty.toFloat()
        )
    }

    private fun invertAffine(m: FloatArray): FloatArray {

        val a = m[0]
        val b = m[1]
        val tx = m[2]

        val c = m[3]
        val d = m[4]
        val ty = m[5]

        val det = a * d - b * c

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
