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

    /*
     * DeepFaceLab FULL face padding.
     */
    private const val DFL_PADDING = 0.2109375f

    data class AlignedFace(
        val bitmap: Bitmap,
        val forward: FloatArray,
        val inverse: FloatArray
    )

    fun align(
        source: Bitmap,
        face: YuNetDetector.Face
    ): AlignedFace {

        require(source.width > 0 && source.height > 0) {
            "Invalid source bitmap"
        }

        /*
         * YuNet landmark coordinates are in the
         * original image coordinate system.
         */
        val src = arrayOf(
            Point2(
                face.rightEye.x,
                face.rightEye.y
            ),
            Point2(
                face.leftEye.x,
                face.leftEye.y
            ),
            Point2(
                face.nose.x,
                face.nose.y
            ),
            Point2(
                face.rightMouth.x,
                face.rightMouth.y
            ),
            Point2(
                face.leftMouth.x,
                face.leftMouth.y
            )
        )

        val dst = canonicalPoints()

        val transform =
            estimateSimilarityTransform(
                src,
                dst
            )

        val inverse =
            invertAffine(transform)

        val matrix = Matrix()

        matrix.setValues(
            floatArrayOf(
                transform[0],
                transform[1],
                transform[2],
                transform[3],
                transform[4],
                transform[5],
                0f,
                0f,
                1f
            )
        )

        val aligned =
            Bitmap.createBitmap(
                SIZE,
                SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(aligned)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            source,
            matrix,
            paint
        )

        return AlignedFace(
            bitmap = aligned,
            forward = transform,
            inverse = inverse
        )
    }

    private fun canonicalPoints(): Array<Point2> {

        val eyeY = 43.0f
        val eyeLX = 39.0f
        val eyeRX = 89.0f

        val noseX = 64.0f
        val noseY = 65.0f

        val mouthLX = 45.0f
        val mouthRX = 83.0f
        val mouthY = 88.0f

        val centerX = 64.0f
        val centerY = 64.0f

        val scale =
            1.0f /
                (1.0f + DFL_PADDING * 2.0f)

        fun expand(
            x: Float,
            y: Float
        ): Point2 {

            return Point2(
                centerX +
                    (x - centerX) * scale,

                centerY +
                    (y - centerY) * scale
            )
        }

        return arrayOf(

            /*
             * YuNet right eye
             */
            expand(
                eyeLX,
                eyeY
            ),

            /*
             * YuNet left eye
             */
            expand(
                eyeRX,
                eyeY
            ),

            /*
             * Nose
             */
            expand(
                noseX,
                noseY
            ),

            /*
             * Right mouth
             */
            expand(
                mouthLX,
                mouthY
            ),

            /*
             * Left mouth
             */
            expand(
                mouthRX,
                mouthY
            )
        )
    }

    private fun estimateSimilarityTransform(
        src: Array<Point2>,
        dst: Array<Point2>
    ): FloatArray {

        require(src.size == 5)
        require(dst.size == 5)

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

            val sx =
                src[i].x.toDouble() -
                    srcCx

            val sy =
                src[i].y.toDouble() -
                    srcCy

            val dx =
                dst[i].x.toDouble() -
                    dstCx

            val dy =
                dst[i].y.toDouble() -
                    dstCy

            a +=
                sx * dx +
                    sy * dy

            b +=
                sx * dy -
                    sy * dx

            denom +=
                sx * sx +
                    sy * sy
        }

        require(denom > 1e-8) {
            "Degenerate face landmarks"
        }

        val scale =
            sqrt(
                a * a +
                    b * b
            ) / denom

        val angle =
            atan2(
                b,
                a
            )

        val c =
            cos(angle) *
                scale

        val s =
            sin(angle) *
                scale

        val tx =
            dstCx -
                (
                    c * srcCx -
                        s * srcCy
                    )

        val ty =
            dstCy -
                (
                    s * srcCx +
                        c * srcCy
                    )

        return floatArrayOf(

            c.toFloat(),
            (-s).toFloat(),
            tx.toFloat(),

            s.toFloat(),
            c.toFloat(),
            ty.toFloat()
        )
    }

    private fun invertAffine(
        m: FloatArray
    ): FloatArray {

        val a = m[0]
        val b = m[1]
        val tx = m[2]

        val c = m[3]
        val d = m[4]
        val ty = m[5]

        val determinant =
            a * d -
                b * c

        require(
            kotlin.math.abs(
                determinant
            ) > 1e-8f
        ) {
            "Non-invertible face transform"
        }

        val invA =
            d /
                determinant

        val invB =
            -b /
                determinant

        val invC =
            -c /
                determinant

        val invD =
            a /
                determinant

        val invTx =
            -(
                invA * tx +
                    invB * ty
                )

        val invTy =
            -(
                invC * tx +
                    invD * ty
                )

        return floatArrayOf(

            invA,
            invB,
            invTx,

            invC,
            invD,
            invTy
        )
    }

    private data class Point2(
        val x: Float,
        val y: Float
    )
}
